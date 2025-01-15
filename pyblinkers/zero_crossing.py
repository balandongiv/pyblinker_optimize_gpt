# LLMed on 15 January 2025

import warnings
import numpy as np

def get_line_intersection_slope(xIntersect, yIntersect, leftXIntercept, rightXIntercept):
    """
    Original logic retained. Computes slopes at the intersection point.
    """
    # Local variable usage here is minimal since there's only two lines.
    leftSlope = yIntersect / (xIntersect - leftXIntercept)
    rightSlope = yIntersect / (xIntersect - rightXIntercept)
    return leftSlope, rightSlope


def get_average_velocity(pLeft, pRight, xLeft, xRight):
    """
    Original logic retained. Computes average velocities.
    """
    # Using local references is possible, but it's already short.
    averLeftVelocity = pLeft.coef[1] / np.std(xLeft)
    averRightVelocity = pRight.coef[1] / np.std(xRight)
    return averLeftVelocity, averRightVelocity


def left_right_zero_crossing(candidateSignal, maxFrame, outerStarts, outerEnds):
    """
    Identify the left zero crossing and right zero crossing of the signal
    between outerStarts->maxFrame and maxFrame->outerEnds.
    """
    startIdx = int(outerStarts)
    mFrame = int(maxFrame)
    endIdx = int(outerEnds)

    # Left side search
    leftRange = np.arange(startIdx, mFrame)
    leftValues = candidateSignal[leftRange]
    sInd_leftZero = np.flatnonzero(leftValues < 0)

    if sInd_leftZero.size > 0:
        leftZero = leftRange[sInd_leftZero[-1]]
    else:
        # Fall back if no negative crossing found in leftRange
        fullLeftRange = np.arange(0, mFrame).astype(int)
        leftNegIdx = np.flatnonzero(candidateSignal[fullLeftRange] < 0)
        leftZero = fullLeftRange[leftNegIdx[-1]]

    # Right side search
    rightRange = np.arange(mFrame, endIdx)
    rightValues = candidateSignal[rightRange]
    sInd_rightZero = np.flatnonzero(rightValues < 0)

    if sInd_rightZero.size > 0:
        rightZero = rightRange[sInd_rightZero[0]]
    else:
        # Extreme remedy by extending beyond outerEnds to the max signal length
        try:
            extremeOuter = np.arange(mFrame, candidateSignal.shape[0]).astype(int)
        except TypeError:
            print('Error')
            # If this except triggers, raise or handle accordingly
            return leftZero, None

        sInd_rightZero_ex = np.flatnonzero(candidateSignal[extremeOuter] < 0)
        if sInd_rightZero_ex.size > 0:
            rightZero = extremeOuter[sInd_rightZero_ex[0]]
        else:
            return leftZero, None

    if leftZero > mFrame:
        raise ValueError("Validation error: leftZero = {leftZero}, maxFrame = {maxFrame}. Ensure leftZero <= maxFrame.")

    if mFrame > rightZero:
        raise ValueError('Validation error: maxFrame = {maxFrame}, rightZero = {rightZero}. Ensure maxFrame <= rightZero.')

    return leftZero, rightZero


def get_up_down_stroke(maxFrame, leftZero, rightZero):
    """
    YES Compute the place of maximum positive and negative velocities.
    upStroke is the interval between leftZero and maxFrame,
    downStroke is the interval between maxFrame and rightZero.
    """
    # Using local references for clarity.
    mFrame = int(maxFrame)
    lZero = int(leftZero)
    rZero = int(rightZero)

    upStroke = np.arange(lZero, mFrame + 1)
    downStroke = np.arange(mFrame, rZero + 1)
    return upStroke, downStroke


def _maxPosVelFrame(blinkVelocity, maxFrame, leftZero, rightZero):
    """
    In the context of *blinkVelocity* time series,
    the `maxPosVelFrame` and `maxNegVelFrame` represent the indices where
    the *blinkVelocity* reaches its maximum positive value and maximum negative value, respectively.
    """
    mFrame = int(maxFrame)
    lZero = int(leftZero)
    rZero = int(rightZero)

    upStroke, downStroke = get_up_down_stroke(mFrame, lZero, rZero)

    # Maximum positive velocity in the upStroke region
    maxPosVelIdx = np.argmax(blinkVelocity[upStroke])
    maxPosVelFrame = upStroke[maxPosVelIdx]

    # Maximum negative velocity in the downStroke region, if it exists
    if downStroke.size > 0:
        maxNegVelIdx = np.argmin(blinkVelocity[downStroke])
        maxNegVelFrame = downStroke[maxNegVelIdx]
    else:
        warnings.warn('Force nan but require further investigation why happen like this')
        maxNegVelFrame = np.nan

    return maxPosVelFrame, maxNegVelFrame


def _get_left_base(blinkVelocity, leftOuter, maxPosVelFrame):
    """
    Determine the left base index from leftOuter to maxPosVelFrame
    by searching for where blinkVelocity crosses <= 0.
    """
    lOuter = int(leftOuter)
    mPosVel = int(maxPosVelFrame)

    leftRange = np.arange(lOuter, mPosVel + 1)
    reversedVelocity = np.flip(blinkVelocity[leftRange])

    leftBaseIndex = np.argmax(reversedVelocity <= 0)
    leftBase = mPosVel - leftBaseIndex - 1
    return leftBase


def _get_right_base(candidateSignal, blinkVelocity, rightOuter, maxNegVelFrame):
    """
    Determine the right base index from maxNegVelFrame to rightOuter
    by searching for where blinkVelocity crosses >= 0.
    """
    rOuter = int(rightOuter)
    mNegVel = int(maxNegVelFrame)

    # Ensure boundaries are valid
    if mNegVel > rOuter:
        return None

    maxSize = candidateSignal.size
    endIdx = min(rOuter, maxSize)
    rightRange = np.arange(mNegVel, endIdx)

    if rightRange.size == 0:
        return None

    # Avoid out-of-bounds indexing for blinkVelocity
    if rightRange[-1] >= blinkVelocity.size:
        rightRange = rightRange[:-1]
        if rightRange.size == 0 or rightRange[-1] >= blinkVelocity.size:
            # TODO: Handle this case more gracefully
            raise ValueError('Please strategies how to address this')

    rightBaseVelocity = blinkVelocity[rightRange]
    rightBaseIndex = np.argmax(rightBaseVelocity >= 0)
    rightBase = mNegVel + rightBaseIndex + 1
    return rightBase


def _get_half_height(candidateSignal, maxFrame, leftZero, rightZero, leftBase, rightOuter):
    """
    leftBaseHalfHeight:
        The coordinate of the signal halfway (in height) between
        the blink maximum and the left base value.
    rightBaseHalfHeight:
        The coordinate of the signal halfway (in height) between
        the blink maximum and the right base value.
    """
    mFrame = int(maxFrame)
    lZero = int(leftZero)
    rZero = int(rightZero)
    lBase = int(leftBase)
    rOuter = int(rightOuter)

    # Halfway point (vertical) from candidateSignal[maxFrame] to candidateSignal[leftBase]
    maxVal = candidateSignal[mFrame]
    leftBaseVal = candidateSignal[lBase]
    halfHeightVal = maxVal - 0.5 * (maxVal - leftBaseVal)

    # Left side half-height from base
    leftRange = np.arange(lBase, mFrame + 1)
    leftVals = candidateSignal[leftRange]
    leftIndex = np.argmax(leftVals >= halfHeightVal)
    leftBaseHalfHeight = lBase + leftIndex + 1

    # Right side half-height from base
    rightRange = np.arange(mFrame, rOuter + 1)
    try:
        rightBaseHalfHeight = min(
            rOuter,
            np.argmax(candidateSignal[rightRange] <= halfHeightVal) + mFrame
        )
    except IndexError:
        # If out-of-bounds, reduce range by 1
        rightRange = np.arange(mFrame, rOuter)
        rightBaseHalfHeight = min(
            rOuter,
            np.argmax(candidateSignal[rightRange] <= halfHeightVal) + mFrame
        )

    # Now compute the left and right half-height frames from zero
    # Halfway from candidateSignal[maxFrame] down to 0 (the "zero" crossing region).
    # leftZeroHalfHeight
    zeroHalfVal = 0.5 * maxVal
    leftZeroRange = np.arange(lZero, mFrame + 1)
    leftZeroIndex = np.argmax(candidateSignal[leftZeroRange] >= zeroHalfVal)
    leftZeroHalfHeight = lZero + leftZeroIndex + 1

    # rightZeroHalfHeight
    rightZeroRange = np.arange(mFrame, rZero + 1)
    rightZeroIndex = np.argmax(candidateSignal[rightZeroRange] <= zeroHalfVal)
    rightZeroHalfHeight = min(rOuter, mFrame + rightZeroIndex)

    return leftZeroHalfHeight, rightZeroHalfHeight, leftBaseHalfHeight, rightBaseHalfHeight


def get_left_range(leftZero, maxFrame, candidateSignal, blinkTop, blinkBottom):
    """
    Identify the left blink range based on blinkTop/blinkBottom thresholds
    within candidateSignal.
    """
    lZero = int(leftZero)
    mFrame = int(maxFrame)

    blinkRange = np.arange(lZero, mFrame + 1, dtype=int)
    candSlice = candidateSignal[blinkRange]

    # Indices where candidateSignal < blinkTop
    topIdx = np.where(candSlice < blinkTop)[0]
    blinkTopPoint_idx = topIdx[-1]  # the last occurrence

    # Indices where candidateSignal > blinkBottom
    bottomIdx = np.flatnonzero(candSlice > blinkBottom)
    blinkBottomPoint_idx = bottomIdx[0]  # the first occurrence

    blinkTopPoint_l_X = blinkRange[blinkTopPoint_idx]
    blinkTopPoint_l_Y = candidateSignal[blinkTopPoint_l_X]

    blinkBottomPoint_l_X = blinkRange[blinkBottomPoint_idx]
    blinkBottomPoint_l_Y = candidateSignal[blinkBottomPoint_l_X]

    leftRange = [blinkBottomPoint_l_X, blinkTopPoint_l_X]

    return leftRange, blinkTopPoint_l_X, blinkTopPoint_l_Y, blinkBottomPoint_l_X, blinkBottomPoint_l_Y


def get_right_range(maxFrame, rightZero, candidateSignal, blinkTop, blinkBottom):
    """
    Identify the right blink range based on blinkTop/blinkBottom thresholds
    within candidateSignal.
    """
    mFrame = int(maxFrame)
    rZero = int(rightZero)

    blinkRange = np.arange(mFrame, rZero + 1, dtype=int)
    candSlice = candidateSignal[blinkRange]

    # Indices where candidateSignal < blinkTop
    topMask = (candSlice < blinkTop)
    blinkTopPoint_r = np.argmax(topMask)  # first True

    # Indices where candidateSignal > blinkBottom
    bottomMask = (candSlice > blinkBottom)
    bottomIdx = np.where(bottomMask)[0]
    blinkBottomPoint_r = bottomIdx[-1]  # last True

    blinkTopPoint_r_X = blinkRange[blinkTopPoint_r]
    blinkTopPoint_r_Y = candidateSignal[blinkTopPoint_r_X]

    blinkBottomPoint_r_X = blinkRange[blinkBottomPoint_r]
    blinkBottomPoint_r_Y = candidateSignal[blinkBottomPoint_r_X]

    rightRange = [blinkRange[blinkTopPoint_r], blinkRange[blinkBottomPoint_r]]

    return (rightRange,
            blinkTopPoint_r_X, blinkTopPoint_r_Y,
            blinkBottomPoint_r_X, blinkBottomPoint_r_Y)


def compute_fit_range(candidateSignal, maxFrame, leftZero, rightZero, baseFraction, top_bottom=None):
    """
    Computes xLeft, xRight, leftRange, rightRange,
    plus optional top/bottom blink points,
    for the candidateSignal around a blink event.
    """
    mFrame = int(maxFrame)
    lZero = int(leftZero)
    rZero = int(rightZero)

    # Compute the blinkTop/blinkBottom for thresholding
    blinkHeight = candidateSignal[mFrame] - candidateSignal[lZero]
    blinkTop = candidateSignal[mFrame] - baseFraction * blinkHeight
    blinkBottom = candidateSignal[lZero] + baseFraction * blinkHeight

    (leftRange,
     blinkTopPoint_l_X, blinkTopPoint_l_Y,
     blinkBottomPoint_l_X, blinkBottomPoint_l_Y) = get_left_range(lZero, mFrame, candidateSignal, blinkTop, blinkBottom)

    (rightRange,
     blinkTopPoint_r_X, blinkTopPoint_r_Y,
     blinkBottomPoint_r_X, blinkBottomPoint_r_Y) = get_right_range(mFrame, rZero, candidateSignal, blinkTop, blinkBottom)

    # Create arrays for fitting
    xLeft = np.arange(leftRange[0], leftRange[1] + 1, dtype=int)  # +1 to include the last index
    xRight = np.arange(rightRange[0], rightRange[1] + 1, dtype=int)

    # Edge-case warnings
    # Edge-case warnings
    if blinkBottomPoint_l_X == blinkTopPoint_l_X:
        warnings.warn(f'same value for left top_blink and left bottom_blink: {blinkBottomPoint_l_X}')

    if blinkBottomPoint_r_X == blinkTopPoint_r_X:
        warnings.warn(f'same value for right top_blink and right bottom_blink: {blinkBottomPoint_r_X}')


    # Replace empty arrays with np.nan for consistency
    if xLeft.size == 0:
        xLeft = np.nan
    if xRight.size == 0:
        xRight = np.nan

    if top_bottom is None:
        # Return minimal information
        warnings.warn('To modify this so that all function return the top_bottom point')
        return xLeft, xRight, leftRange, rightRange
    else:
        # Return extended info including top/bottom points
        return (xLeft, xRight, leftRange, rightRange,
                blinkBottomPoint_l_Y, blinkBottomPoint_l_X,
                blinkTopPoint_l_Y, blinkTopPoint_l_X,
                blinkBottomPoint_r_X, blinkBottomPoint_r_Y,
                blinkTopPoint_r_X, blinkTopPoint_r_Y)


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")

def viz_complete_blink_prop(data,row,srate):


    """

    TODO Viz

    https://stackoverflow.com/a/51928241/6446053

    :return:
    """


    xLabelString='T'
    fig, ax = plt.subplots(figsize=(8, 6))

    npad = 20
    preLimit = row['startBlinks'] - npad
    postLimit = row['endBlinks'] + npad

    idx_t = np.arange(preLimit, postLimit + 1)

    bTrace = data[idx_t]


    plt.plot(idx_t, bTrace,linestyle='-',marker='o',color='b',
             label='line with marker',alpha=0.7)
    plt.plot([idx_t[0], idx_t[-1]], [0, 0], "--", color="gray", lw=2,label='Y0')


    plt.plot([row['xLineCross_l'] , row['xIntersect']], [row['yLineCross_l'],  row['yIntersect']], "--", color="gray", lw=2)
    plt.plot([row['xIntersect'],row['xLineCross_r']], [row['yIntersect'],row['yLineCross_r']], "--", color="gray", lw=2)

    ## PLot key point
    plt.scatter([row['blinkBottomPoint_l_X'],row['blinkTopPoint_l_X']],
                [row['blinkBottomPoint_l_Y'],row['blinkTopPoint_l_Y']],
                marker='*', s=200,label='left_top_down_blink')

    plt.scatter([row['blinkBottomPoint_r_X'],row['blinkTopPoint_r_X']],
                [row['blinkBottomPoint_r_Y'],row['blinkTopPoint_r_Y']],
                marker='*', s=200,label='right_top_down_blink')



    plt.scatter(row['xIntersect'], row['yIntersect'],label='tent_point')


    plt.scatter([row['leftZero'], row['rightZero']], [0, 0], marker='d', s=100,label='zero crossing')
    plt.scatter(row['maxFrame'], data[row['maxFrame']],label='max Frame')

    plt.scatter([row['leftBase'], row['rightBase']],
                [data[row['leftBase']], data[row['rightBase']]],label='base')

    plt.scatter([row['leftBaseHalfHeight'], row['rightBaseHalfHeight']],
                [data[row['leftBaseHalfHeight']], data[row['rightBaseHalfHeight']]],
                marker='<', s=200,label='BaseHalfHeight')

    plt.scatter([row['rightZeroHalfHeight'], row['leftZeroHalfHeight']],
                [data[row['rightZeroHalfHeight']], data[row['leftZeroHalfHeight']]],
                marker='>', s=300,label='ZeroHalfHeight')

    plt.scatter([row['maxPosVelFrame'],  row['maxNegVelFrame']],
                [data[row['maxPosVelFrame']], data[ row['maxNegVelFrame']]],
                marker='>', s=300,label='maxVelFrame')



    plt.legend()
    ylabel='Signal(uv)'
    plt.xlabel(xLabelString)
    plt.ylabel(ylabel)
    bquality=row['blink_quality'] == 'Good'
    maxFrames=row['maxFrames']
    d=dict(fig=fig,blink_quality=bquality,maxFrames=maxFrames)

    return d
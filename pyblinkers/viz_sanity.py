

# import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt

def viz_line_intersection(candidateSignal,xRight,xLeft,left_x_intercept,right_x_intercept,
                          x_intersect,y_intersect):
    #### make_Plot


    srate=100
    npad=20
    idx_t=(np.arange ( np.min(xLeft)-npad, np.max(xRight) +npad))
    bTrace= candidateSignal[idx_t]
    t=idx_t
    #
    plt.plot(t, bTrace, linewidth=10,alpha=0.5)


    plt.plot([left_x_intercept[0],x_intersect[0]], [0,y_intersect[0]],'--')


    plt.plot([x_intersect[0],right_x_intercept[0],], [y_intersect[0],0],'--')
    # plt.show()


    z=[left_x_intercept,right_x_intercept]
    y=[0,0]

    n =  [f'XIntercept_{float("{0:.2f}".format(i[0]))}' for i in z]
    plt.scatter(z,y)

    for label, x, y in zip(n, z,y):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))


    plt.scatter([x_intersect],[y_intersect])

    for label, x, y in zip([f'xy_intersect'], [x_intersect],[y_intersect]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    plt.show()

def viz_blink_top_buttom_point(candidateSignal,blinkRange,blinkTop,blinkBottom,max_blink,right_zero,leftRange,rightRange):

    srate=1
    npad=20
    idx_t=(np.arange ( max_blink-npad, right_zero +npad))
    bTrace= candidateSignal[idx_t]
    t=idx_t/srate
    #
    plt.plot(t, bTrace)
    #
    maxFrame_Y=candidateSignal[max_blink]
    rightZero_Y=candidateSignal[right_zero]
    ee=[(max_blink)/srate,(right_zero)/srate]
    ddd=[maxFrame_Y,rightZero_Y]
    #
    plt.scatter(ee,ddd)



    leftRange_x=[leftRange[0]/srate,leftRange[1]/srate]

    plt.scatter(leftRange_x,[0,0])

    rightRange_x=[rightRange[0]/srate,rightRange[1]/srate]

    plt.scatter(rightRange_x,[0,0])

    plt.show()
    hh=1


def cross_check_maxFrame(candidateSignal,max_blink,srate):

    idx_t=(np.arange ( max_blink-10, max_blink +10))
    bTrace= candidateSignal[idx_t]
    t=idx_t/srate

    plt.plot(t, bTrace)

    ddd=candidateSignal[max_blink-1]
    ee=(max_blink-1)/srate

    plt.scatter(ee,ddd)

    plt.show()

def _viz_sanity_zero_crossing(start_f,stop_f,max_blink,dsignal,dlegend=''):

    srate=1
    npad=20
    preLimit=start_f-npad
    postLimit=stop_f+npad
    idx_t=np.arange ( preLimit, postLimit)

    t=idx_t/srate
    bTrace =dsignal [idx_t]
    blinkScale = np.max(dsignal [np.arange ( preLimit-1, postLimit )])
    bTrace = bTrace*blinkScale/max(bTrace)
    plt.plot(t, bTrace)
    plt.scatter([start_f / srate, stop_f / srate],
                               [dsignal[start_f], dsignal[stop_f]],
                               marker='*',s=300)

    plt.scatter([max_blink],
                [dsignal[max_blink]],
                marker='o',s=300)

    plt.plot([t[0], t[-1]], [0, 0], 'r-', lw=2)
    plt.savefig('zero_Crossing_new_01.png')

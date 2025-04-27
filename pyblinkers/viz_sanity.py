

# import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt
def plot_epoch_signal(epoch_data, global_idx, ch_idx, sampling_rate=None):
    """
    Plots the signal from a specific epoch and channel.

    Parameters:
    - epoch_data: 3D array of shape (n_epochs, n_channels, n_times)
    - global_idx: int, index of the epoch to plot
    - ch_idx: int, index of the channel to plot
    - sampling_rate: float or int, optional. If provided, x-axis will be in seconds
    """
    epoch_signal = epoch_data[global_idx, ch_idx, :]
    n_times = epoch_signal.shape[0]

    if sampling_rate:
        time = [i / sampling_rate for i in range(n_times)]
        xlabel = 'Time (s)'
    else:
        time = range(n_times)
        xlabel = 'Time (samples)'

    plt.figure(figsize=(10, 4))
    plt.plot(time, epoch_signal)
    plt.title(f'Epoch Signal - Epoch: {global_idx}, Channel: {ch_idx}')
    plt.xlabel(xlabel)
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('test.png')
    # plt.show()
    v=1
def viz_line_intersection(candidateSignal,xRight,xLeft,leftXIntercept,rightXIntercept,
                          xIntersect,yIntersect):
    #### make_Plot


    srate=100
    npad=20
    idx_t=(np.arange ( np.min(xLeft)-npad, np.max(xRight) +npad))
    bTrace= candidateSignal[idx_t]
    t=idx_t
    #
    plt.plot(t, bTrace, linewidth=10,alpha=0.5)


    plt.plot([leftXIntercept[0],xIntersect[0]], [0,yIntersect[0]],'--')


    plt.plot([xIntersect[0],rightXIntercept[0],], [yIntersect[0],0],'--')
    # plt.show()


    z=[leftXIntercept,rightXIntercept]
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


    plt.scatter([xIntersect],[yIntersect])

    for label, x, y in zip([f'xy_intersect'], [xIntersect],[yIntersect]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    plt.show()

def viz_blink_top_buttom_point(candidateSignal,blinkRange,blinkTop,blinkBottom,maxFrame,rightZero,leftRange,rightRange):

    srate=1
    npad=20
    idx_t=(np.arange ( maxFrame-npad, rightZero +npad))
    bTrace= candidateSignal[idx_t]
    t=idx_t/srate
    #
    plt.plot(t, bTrace)
    #
    maxFrame_Y=candidateSignal[maxFrame]
    rightZero_Y=candidateSignal[rightZero]
    ee=[(maxFrame)/srate,(rightZero)/srate]
    ddd=[maxFrame_Y,rightZero_Y]
    #
    plt.scatter(ee,ddd)



    leftRange_x=[leftRange[0]/srate,leftRange[1]/srate]

    plt.scatter(leftRange_x,[0,0])

    rightRange_x=[rightRange[0]/srate,rightRange[1]/srate]

    plt.scatter(rightRange_x,[0,0])

    plt.show()
    hh=1


def cross_check_maxFrame(candidateSignal,maxFrame,srate):

    idx_t=(np.arange ( maxFrame-10, maxFrame +10))
    bTrace= candidateSignal[idx_t]
    t=idx_t/srate

    plt.plot(t, bTrace)

    ddd=candidateSignal[maxFrame-1]
    ee=(maxFrame-1)/srate

    plt.scatter(ee,ddd)

    plt.show()

def _viz_sanity_zero_crossing(start_f,stop_f,maxFrame,dsignal,dlegend=''):

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

    plt.scatter([maxFrame],
                [dsignal[maxFrame]],
                marker='o',s=300)

    plt.plot([t[0], t[-1]], [0, 0], 'r-', lw=2)
    plt.savefig('zero_Crossing_new_01.png')

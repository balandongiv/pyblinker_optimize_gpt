import warnings

import hickle as hkl
import numpy as np
import pandas as pd

# d=dict(pk=[4,8,11,15,20,24],lf=[4,7,3,15,19,20])
# df=pd.DataFrame(d)
# data=np.arange(2,60,2)
# df['y_pk']=data[df['pk']]
# df['y_lf']=data[df['lf']]
# df['a']=data[df['pk']]-data[df['lf']]
# h=1
def _left_range(arr,st,en,blinkTop_th,blinkBottom_th):
    warnings.warn('To find better way converting both integers at 1 go')

    data_slice=arr[np.arange(int(st),int(en))]

    blinkTopPoint=np.argmin(data_slice<blinkTop_th)
    blinkBottomPoint=np.argmax(data_slice>blinkBottom_th)

    ## These two values represent xLeft
    leftRange_start=arr[blinkBottomPoint]
    leftRange_end=arr[blinkTopPoint]
    return blinkTopPoint,blinkBottomPoint,leftRange_start,leftRange_end

def _right_range(arr,st,en,blinkTop_th,blinkBottom_th):
    warnings.warn('To find better way converting both integers at 1 go')

    data_slice=arr[np.arange(int(st),int(en))]

    blinkTopPoint=np.argmin(data_slice<blinkTop_th)
    blinkBottomPoint=np.argmax(data_slice>blinkBottom_th)

    ## These two values represent xRight
    rightRange_start=arr[blinkTopPoint]
    rightRange_end=arr[blinkBottomPoint]
    return blinkTopPoint,blinkBottomPoint,rightRange_start,rightRange_end


# blinkTopPoint_ = np.argmax(candidateSignal[blinkRange] < blinkTop)
#
# blinkBottomPoint_ = np.argmin(candidateSignal[blinkRange] > blinkBottom)
# rightRange = [blinkRange[blinkTopPoint_], blinkRange[blinkBottomPoint_]]

filename = 'get_top_bottom_position.hkl'
# hkl.dump([blinkComp, df], filename)
# njob=1
# raw=1
baseFraction = 0.1  # Fraction from top and bottom
data, df=hkl.load(filename)


df['blinkHeight'] = data[df['peaks_point']] - data[df['zero_left']]  # ?? 8.8286028

df['blinkTop'] = data[df['peaks_point']]  - baseFraction * df['blinkHeight']  # ?? 7.8665667

df['blinkBottom'] = data[df['zero_left']]  - baseFraction * df['blinkHeight']  # ?? 7.8665667

warnings.warn('Please set type as int for both zero_left and peaks_points')

df[['blinkTopPoint','blinkBottomPoint','leftRange_start','leftRange_end']]=\
    df.apply(lambda x: _left_range(data,x['zero_left'],x['peaks_point'],x['blinkTop'],x['blinkBottom']),
             axis=1,result_type="expand")

df['leftRange_start']
# TODO NEED TO VISUALISE WHAT ACTUALLY THIS NUMBER (blinkTopPoint and blinkBottomPoint) mean

df[['blinkTopPoint_l','blinkBottomPoint_','leftRange_start','leftRange_end']]= \
    df.apply(lambda x: _left_range(data,x['zero_left'],x['peaks_point'],x['blinkTop'],x['blinkBottom']),
             axis=1,result_type="expand")

df[['blinkTopPoint_r','blinkBottomPoint_l','rightRange_start','rightRange_end']]= \
    df.apply(lambda x: _right_range(data,x['peaks_point'],x['zero_right'],x['blinkTop'],x['blinkBottom']),
             axis=1,result_type="expand")


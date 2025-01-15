import numpy as np
import pandas as pd


def vectorise_slicing():
    """

    Please get ready for potential error
    :return:
    """
    d=dict(pk=[4,8,11,15,20,24],lf=[4,7,3,15,19,20])
    df=pd.DataFrame(d)
    data=np.arange(2,60,2)
    df['y_pk']=data[df['pk']]
    df['y_lf']=data[df['lf']]
    df['a']=data[df['pk']]-data[df['lf']]

def fx(arr,st,en,blinkTop_th,blinkBottom_th):
    data_slice=arr[np.arange(st,en)]
    return np.argmin(data_slice<blinkTop_th),np.argmax(data_slice>blinkBottom_th)


def vectorise_slicing_continous():
    """

    Please get ready for potential error
    :return:
    """
    th=0.9
    np.random.seed(0)

    arr=np.array([.1,.11,.21,.01,.5,.7,.91,.92,.95,  # 8 select 6 range:1-12
                  .96,.1,.21,.23,.6,.7,.71,.72,.95,0.96,0.97])     # select 15 range 10-17
    # d=dict(s=[1,10],e=[12,19],doutputs=[5,7])
    d=dict(s=[1,10],e=[12,19])
    df=pd.DataFrame(dict(s=[1,10],e=[12,19]))

    df[['blinkTopPoint','blinkBottomPoint']]=df.apply(lambda x: fx(arr,x['s'],x['e'],th),
                                                      axis=1,result_type="expand")
    ##
    drange=np.arange(1,12)
    rr=arr[drange]
    dd=rr<th
    doutput = np.argmin(dd)
    lr_bbt=drange[doutput]

    drange=np.arange(10,19)
    rr=arr[drange]
    dd=rr<th
    doutput = np.argmin(dd)

    leftRange = [blinkRange[blinkBottomPoint], blinkRange[blinkTopPoint]]

    df['y_pk']=data[df['pk']]
    df['y_lf']=data[df['lf']]
    df['a']=data[df['pk']]-data[df['lf']]

vectorise_slicing_continous()
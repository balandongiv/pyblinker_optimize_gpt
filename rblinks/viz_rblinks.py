
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def viz_peak_zero_crossing(data,df):

    warnings.warn('Dropping row with nan. It is worth to maintain should you want to investigate')
    df=df.dropna().reset_index(drop=True)
    sns.set_style("darkgrid")
    h=1
    leftp=df['zero_left'].to_numpy().astype(int)
    rightp=df['zero_right'].to_numpy().astype(int)
    base_left=df['base_left'].to_numpy().astype(int)
    base_right=df['base_right'].to_numpy().astype(int)
    peaks=df['peaks_point'].to_numpy().astype(int)
    #
    #
    # fig, ax = plt.subplots()
    plt.plot(data)
    plt.plot(peaks, data[peaks], "x",label='Local Maxima')
    # sns.scatterplot(data=df, x="total_bill", y="tip")
    #
    plt.scatter(leftp, data[leftp],marker='x',s=100,alpha=0.7,label='leading_valleys')
    plt.scatter(rightp,data[rightp],marker='x',s=100,alpha=0.7,label='Trailing_valleys')

    plt.scatter(base_right,data[base_right],marker='o',s=100,alpha=0.7,label='base_right')
    plt.scatter(base_left,data[base_left],marker='o',s=100,alpha=0.7,label='base_left')
    #
    # ax.scatter(x=df["x"], y=df["y"], c=df["val"])
    plt.plot(np.zeros_like(data), "--", color="gray")
    plt.legend()
    plt.show()
    j=1
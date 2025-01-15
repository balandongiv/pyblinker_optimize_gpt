import numpy as np

'''

The median absolute deviation (MAD) is defined as:

MAD
=
median
(
∣
𝑥
𝑖
−
median
(
𝑥
)
∣
)
MAD=median(∣x 
i
​
 −median(x)∣)
For a normal distribution, the MAD is related to the standard deviation by a fixed scaling factor. Specifically:

MAD
≈
𝜎
Φ
−
1
(
0.75
)
MAD≈ 
Φ 
−1
 (0.75)
σ
​
 
Here:

𝜎
σ is the standard deviation of the normal distribution.
Φ
−
1
(
0.75
)
Φ 
−1
 (0.75) is the 75th percentile of the standard normal distribution, which is approximately 0.6745.
Thus, the scaling factor is:

1
Φ
−
1
(
0.75
)
=
1
0.6745
≈
1.4826
Φ 
−1
 (0.75)
1
​
 = 
0.6745
1
​
 ≈1.4826
Purpose of Using 1.4826
Using the factor 1.4826 makes the MAD a robust estimator of the standard deviation in the presence of outliers, as the MAD is less sensitive to extreme values than the standard deviation. This property makes it widely used in robust statistical methods.

Practical Use
When you compute:

Robust Standard Deviation
=
1.4826
×
MAD
Robust Standard Deviation=1.4826×MAD
you obtain an estimate of the standard deviation that is resistant to outliers, making it especially useful in fields like signal processing, finance, and machine learning.

'''
scalingFactor = 1.4826

params = {'stdThreshold': 1.50,
          'minEventLen': 0.05,
          'minEventSep': 0.05,
          'baseFraction': 0.1,
          'correlationThresholdTop': 0.980,
          'correlationThresholdBottom': 0.90,
          'correlationThresholdMiddle': 0.95,
          'shutAmpFraction': 0.9,
          'blinkAmpRange_1': 3,
          'blinkAmpRange_2': 50,
          'goodRatioThreshold': 0.7,
          'minGoodBlinks': 10,
          'keepSignals': 0,
          'correlationThreshold': 0.98,
          'pAVRThreshold': 3,
          'zThresholds':np.array([[0.9, 0.98], [2.0, 5.0]]),
          'sfreq': 100}



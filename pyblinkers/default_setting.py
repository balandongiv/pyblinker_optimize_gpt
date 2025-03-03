import numpy as np


SCALING_FACTOR= 1.4826 # From original paper: by default, BLINKER eliminates “best blinks” more than five robust standard deviations from the median and “good” blinks more than two robust standard deviations away from this median. Here we define the robust standard deviation as 1.4826 times the median absolute deviation from the median. Figure 3 displays the median of the “best” blinks with a gray vertical line and the locations that are two robust standard deviations from this median with dashed gray lines.

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
          'pAVRThreshold': 3, # from original paper: The pAVR criterion captures the difference between the sharp rising edge of saccades and the more curved rise of normal blinks. We have found empirically that blink candidates with pAVR ≤ 3 do not correspond to normal blinks, but rather saccades having short, fast eye movements
          'z_thresholds':np.array([[0.9, 0.98], [2.0, 5.0]]),
          'sfreq': 100}



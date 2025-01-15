import numpy as np

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



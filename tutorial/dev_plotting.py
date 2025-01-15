import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing._peak_finder import peak_finder
import matplotlib.pyplot as plt

"""
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
"""
t = np.arange(0, 3, 0.01)
y = np.sin(np.pi*t) - np.sin(0.5*np.pi*t)
idx2 = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0] + 1

# plt.plot(t, y)
# plt.show()
# peak_locs, peak_mags = peak_finder(y) # doctest: +SKIP

tnew=t[idx2]
ynew=y[idx2]
# plt.plot(tnew, peak_mags, 'r-', lw=2)
plt.scatter(tnew, ynew)
plt.show()

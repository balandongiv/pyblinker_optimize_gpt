'''

ValueError: attempt to get argmax of an empty sequence
'''

import numpy as np
import mne
from pyblinkers.pyblinker import BlinkDetector
raw = mne.io.read_raw_fif("/home/rpb/IdeaProjects/pyblinkers/debug_pyblinker.fif")
# arr=np.arange(1,20,1)
# g=arr[5:15][2:-1]
# hh=np.argmax(g <10)
# aaa=[1,2,3][6,7,8]
annot, ch, number_good_blinks,_ = BlinkDetector(raw,visualize=False, annot_label=None).get_blink_stat()
raw.set_annotations(annot)
raw.plot(block=True, title=f'Eye close based on channel {ch}')


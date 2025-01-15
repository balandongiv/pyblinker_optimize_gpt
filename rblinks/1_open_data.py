import mne

from pyblinkers import default_setting
from rblinks.find_valley_in import get_peak_base_zero_crossing
j=1
###
raw=mne.io.read_raw_fif('raw_audvis_resampled.fif')
params = default_setting.params
blinkComp=raw.get_data(picks='EEG 017')[0]
sfreq = raw.info['sfreq']


df=get_peak_base_zero_crossing(blinkComp, height=0)

import hickle as hkl
filename = 'get_top_bottom_position.hkl'
hkl.dump([blinkComp, df], filename)
# njob=1
# raw=1
# all_result=hkl.load(filename)
# signalData = extracBlinks(all_result, raw, params, njob)
# filename = 'create_signalData.hkl'
# hkl.dump(signalData, filename)
# signalData=hkl.load(filename)

h=1


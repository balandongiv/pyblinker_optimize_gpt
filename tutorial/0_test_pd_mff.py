
import mne

from pyblinkers.pyblinker import BlinkDetector
fname =r'C:\Users\balan\OneDrive\Desktop\dataset\drowsy_driving_raja\S6\P.mff'

raw = mne.io.read_raw_egi(fname, preload=True)
raw.filter(0.5, 20.5, fir_design='firwin')
ch_frontal = ['E1', 'E2', 'E3', 'E8', 'E9', 'E10', 'E14', 'E15', 'E16', 'E17', 'E18']
montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
montage.ch_names[-1] = 'VREF'
raw.set_montage(montage, match_case=False)


cathode = ['E126', 'E127', 'E1', 'E32']
anode = ['E8', 'E25', 'E17', 'E17']
virtual_name = ['eog_vert_right', 'eog_vert_left', 'eog_hor_right', 'eog_hor_left']
raw = mne.set_bipolar_reference(raw, anode, cathode, ch_name=virtual_name, drop_refs=False)
raw.set_channel_types({i: 'eog' for i in virtual_name})
raw.set_channel_types({i: 'eeg' for i in ch_frontal})
raw = raw.pick_types(eog=True)
annot, ch, number_good_blinks, df = BlinkDetector(raw, visualize=False).get_blink_stat()

print(annot)
raw.set_annotations(annot)
import matplotlib
matplotlib.use('TkAgg')
raw.plot(block=True,title=f'Eye close based on channel {Ch}')

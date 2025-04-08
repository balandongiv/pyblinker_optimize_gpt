
from pyblinkers.utils._logging import _pbar
interp_channels=[1,2,3]
verbose=True
pos=2
for epoch_idx, interp_chs in _pbar(
        list(enumerate(interp_channels)),
        desc='Repairing epochs',
        position=pos, leave=True, verbose=verbose):
    g=12
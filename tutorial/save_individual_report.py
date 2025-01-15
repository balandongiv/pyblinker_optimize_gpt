import os

import hickle as hkl
import matplotlib
import mne

matplotlib.use('Agg')
from pyblinkers.utilities.misc import check_make_folder
from tqdm import tqdm

def save_blink_report(dlist):
    rpath = os.getcwd()
    froot = os.path.join(rpath, 'btutorial')
    check_make_folder(froot)
    spath = os.path.join(froot, 'mff_sbj_17_reportsss.html')
    save_type = 'all_good'

    if save_type == 'all_good':
        dlist = list(filter(lambda dlist: dlist['blink_quality'] == True, dlist))

    title = 'Single page blink summary report'
    rep = mne.Report(title=title)

    for ditem in tqdm(dlist):
        disfig = ditem['fig']
        discaption = str(ditem['maxFrames'])
        section = 'Gblink' if ditem['blink_quality'] else 'Bblink'
        rep.add_figs_to_section(disfig, captions=discaption, section=section)

    rep.save(spath, overwrite=True, open_browser=False)


def save_fig_manual_inspection(dlist):
    rpath = os.getcwd()
    fbad = os.path.join(rpath, 'btutorial', 'bad')
    fgood = os.path.join(rpath, 'btutorial', 'good')
    for ffolder in [fbad, fgood]:
        check_make_folder(ffolder)
        for sub_c in ['0_accept', '0_reject']:
            dpath = os.path.join(ffolder, sub_c)
            check_make_folder(dpath)

    for d in tqdm(dlist):
        dfig = d['fig']
        bquality = d['blink_quality']
        bgq = 'Gblink' if bquality else 'Bblink'
        fsub = fgood if bquality else fbad
        mF = d['maxFrames']
        spath = os.path.join(fsub, f'maxFrames_{mF}_{bgq}.png')

        dfig.savefig(spath)


filename = 'test_gzip.hkl'
dlist = hkl.load(filename)
# save_blink_report(dlist)
save_fig_manual_inspection(dlist)

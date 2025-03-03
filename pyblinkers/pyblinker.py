import logging

import pandas as pd
from tqdm import tqdm

from pyblinkers import default_setting
from pyblinkers.extractBlinkProperties import BlinkProperties, get_good_blink_mask
from pyblinkers.extractBlinkProperties import get_blink_statistic
from pyblinkers.fit_blink import FitBlinks
from pyblinkers.getBlinkPositions import get_blink_position
from pyblinkers.getRepresentativeChannel import channel_selection
from pyblinkers.misc import create_annotation
from pyblinkers.viz_pd import viz_complete_blink_prop

logging.getLogger().setLevel(logging.INFO)


class BlinkDetector:
    def __init__(self, raw_data, visualize=False, annot_label=None, filter_bad=False, filter_low=0.5, filter_high=20.5, resample_rate=100, n_jobs=1, use_multiprocessing=False):
        self.filter_bad = filter_bad
        self.raw_data = raw_data
        self.viz_data = visualize
        self.annot_label = annot_label
        self.sfreq = self.raw_data.info['sfreq']
        self.params = default_setting.params
        self.channel_list = self.raw_data.ch_names
        self.all_data_info = []  # Ensure this line is present
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.resample_rate = resample_rate
        self.n_jobs = n_jobs
        self.use_multiprocessing = use_multiprocessing
        self.all_data=[]

    def prepare_raw_signal(self):
        self.raw_data.pick_types(eeg=True)
        self.raw_data.filter(self.filter_low, self.filter_high, fir_design='firwin', n_jobs=self.n_jobs)
        self.raw_data.resample(self.resample_rate, n_jobs=self.n_jobs)
        return self.raw_data

    def process_channel_data(self, channel):

        # STEP 1: Get blink positions
        df = get_blink_position(self.params, blink_component=self.raw_data.get_data(picks=channel)[0], ch='No_channel')

        # STEP 2: Fit blinks
        fitblinks = FitBlinks(candidate_signal=self.raw_data.get_data(picks=channel)[0], df=df, params=self.params)
        fitblinks.dprocess()
        df = fitblinks.frame_blinks

        # STEP 3: Extract blink statistics
        blink_stats= get_blink_statistic(df, self.params['z_thresholds'], signal=self.raw_data.get_data(picks=channel)[0])
        blink_stats['ch'] = channel
        # STEP 4: Get good blink mask
        good_blink_mask, df = get_good_blink_mask(
            df, blink_stats['bestMedian'], blink_stats['bestRobustStd'], self.params['z_thresholds']
        )

        # STEP 5: Compute blink properties
        df = BlinkProperties(self.raw_data.get_data(picks=channel)[0], df, self.params['sfreq'], self.params).df

        # STEP 6: Apply pAVR restriction
        condition_1 = df['posAmpVelRatioZero'] < self.params['pAVRThreshold']
        condition_2 = df['maxValue'] < (blink_stats['bestMedian'] - blink_stats['bestRobustStd'])
        df = df[~(condition_1 & condition_2)]


        self.all_data_info.append(dict(df=df, ch=channel))
        self.all_data.append(blink_stats)

    @staticmethod
    def filter_point(ch,all_data_info):
        return list(filter(lambda all_data_info: all_data_info['ch'] == ch, all_data_info))[0]


    def filter_bad_blink(self,df):
        # filter_bad = False
        if self.filter_bad:
            df = df[df['blink_quality'] == 'Good']
        return df


    def generate_viz(self,data,df):
        fig_data = [viz_complete_blink_prop(data, row, self.sfreq) for index, row in df.iterrows()]

        return fig_data



    def process_all_channels(self):
        for channel in tqdm(self.channel_list, desc="Processing Channels", unit="channel"):
            self.process_channel_data(channel)

    def select_representative_channel(self):
        ch_blink_stat = pd.DataFrame(self.all_data)
        ch_selected = channel_selection(ch_blink_stat, self.params)
        ch_selected.reset_index(drop=True, inplace=True)
        return ch_selected

    def get_representative_blink_data(self, ch_selected):
        ch = ch_selected.loc[0, 'ch']
        data = self.raw_data.get_data(picks=ch)[0]
        rep_blink_channel = self.filter_point(ch, self.all_data_info)
        df = rep_blink_channel['df']
        df = self.filter_bad_blink(df)
        return ch, data, df

    def create_annotations(self, df):
        annot_description = self.annot_label if self.annot_label else 'eye_blink'
        return create_annotation(df, self.sfreq, annot_description)

    def get_blink(self):
        self.process_all_channels()
        ch_selected = self.select_representative_channel()
        ch, data, df = self.get_representative_blink_data(ch_selected)
        annot = self.create_annotations(df)
        fig_data = self.generate_viz(data, df) if self.viz_data else []
        n_good_blinks = ch_selected.loc[0, 'numberGoodBlinks']
        return annot, ch, n_good_blinks, df, fig_data, ch_selected

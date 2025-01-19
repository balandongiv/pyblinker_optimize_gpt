import logging

import pandas as pd

from pyblinkers import default_setting
from pyblinkers.extractBlinkProperties import BlinkProperties, getGoodBlinkMask
from pyblinkers.extractBlinkProperties import get_blink_statistic
from pyblinkers.fit_blink import FitBlinks
from pyblinkers.misc import create_annotation
from pyblinkers.viz_pd import viz_complete_blink_prop
from pyblinkers.getRepresentativeChannel import ChannelSelection
from pyblinkers.getBlinkPositions_vislab import getBlinkPosition


logging.getLogger().setLevel(logging.INFO)


class BlinkDetector:
    def __init__(self, raw_data, visualize=False, annot_label=None,filter_bad=False):
        self.filter_bad=filter_bad
        self.raw_data = raw_data
        self.viz_data = visualize
        self.annot_label = annot_label
        self.sfreq = self.raw_data.info['sfreq']
        self.params = default_setting.params
        self.channel_list = self.raw_data.ch_names
        self.all_data_info = []
        self.all_data = []

    def prepare_raw_signal(self):
        self.raw_data.pick_types(eeg=True)
        self.raw_data.filter(0.5, 20.5, fir_design='firwin')
        self.raw_data.resample(100)
        return self.raw_data

    def process_channel_data(self, channel):

        # STEP 1: Get blink positions
        df = getBlinkPosition(self.params, blinkComp=self.raw_data.get_data(picks=channel)[0], ch='No_channel')

        # STEP 2: Fit blinks
        fitblinks = FitBlinks(data=self.raw_data.get_data(picks=channel)[0], df=df, params=self.params)
        fitblinks.dprocess()
        df = fitblinks.frame_blinks

        # STEP 3: Extract blink statistics
        blink_stats= get_blink_statistic(df, self.params['zThresholds'], signal=self.raw_data.get_data(picks=channel)[0])
        blink_stats['ch'] = channel
        # STEP 4: Get good blink mask
        good_blink_mask, df = getGoodBlinkMask(
            df, blink_stats['bestMedian'], blink_stats['bestRobustStd'], self.params['zThresholds']
        )

        # STEP 5: Compute blink properties
        sfreq = self.params['sfreq']
        df = BlinkProperties(self.raw_data.get_data(picks=channel)[0], df, sfreq, self.params).df

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
    def get_blink_stat(self):
        for channel in self.channel_list:
            self.process_channel_data(channel)


        ch_blink_stat = pd.DataFrame(self.all_data)
        ch_selected = ChannelSelection(ch_blink_stat, self.params)
        ch_selected.reset_index(drop=True, inplace=True)
        ch = ch_selected.loc[0, 'ch']
        nGoodBlinks = ch_selected.loc[0, 'numberGoodBlinks']
        data = self.raw_data.get_data(picks=ch)[0]
        rep_blink_channel = self.filter_point(ch,self.all_data_info)
        df = rep_blink_channel['df']

        df=self.filter_bad_blink(df)
        # df.to_pickle('unit_test_1.pkl')

        annot_description = self.annot_label if self.annot_label else 'eye_blink'
        annot = create_annotation(df, self.sfreq, annot_description)
        if self.viz_data:
            fig_data=self.generate_viz(data,df)

        else:
            fig_data=[]
        return annot, ch, nGoodBlinks,df,fig_data,ch_selected
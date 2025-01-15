import logging

import pandas as pd

from pyblinkers import default_setting
from pyblinkers.getRepresentativeChannel import filter_blink_amplitude_ratios, filter_good_blinks, filter_good_ratio, \
    select_max_good_blinks
from pyblinkers.utilities.extractBlinkProperties import BlinkProperties
from pyblinkers.utilities.extractBlinkProperties import getGoodBlinkMask
from pyblinkers.utilities.extractBlinkProperties import get_blink_statistic
from pyblinkers.utilities.fit_blink import FitBlinks
from pyblinkers.utilities.misc import create_annotation
from pyblinkers.vislab.getBlinkPositions_vislab import getBlinkPosition
from pyblinkers.viz.viz_pd import viz_complete_blink_prop

logging.getLogger().setLevel(logging.INFO)


class BlinkDetector:
    def __init__(self, raw_data, visualize=False,
                 annot_label=None,filter_bad=False,selected_channels=None):
        self.filter_bad=filter_bad
        if selected_channels:
            self.raw_data = self.getCandidateSignal(raw_data,selected_channels)
        else:
            self.raw_data = raw_data

        self.viz_data = visualize
        self.annot_label = annot_label
        self.sfreq = self.raw_data.info['sfreq']
        self.params = default_setting.params
        self.params['sfreq'] = self.sfreq
        self.channel_list = self.raw_data.ch_names
        self.all_data_info = []
        self.all_data = []
    def getCandidateSignal(self,raw,selected_channels):
        '''

        Extract candidate EEG/EOG channel signals from EEG structure and filter

        Parameters: EEG an EEGLAB EEG structure params parameter structure to override default parameters or struct() for no override (see below for details Output: candidateSignals array of selected signals: numSignals x numFrames signalType type of signal returned: 'ICs', 'SignalNumbers', or 'SignalLabels' signalNumbers positions of candidateSignals in the EEG.data or in the signalLabels cell array of names of candidateSignals params parameter structure with all values filled in.

        The function band-pass filters from 1 Hz to 20 Hz prior to analysis unless params.lowCutoffHz, params.highCutoffHz override (not recommended)

        The function uses channel numbers unless params.signalTypeIndicator has value 'UseICs' or 'UseLabels'.

        '''
        # List of channels to select
        # selected_channels = ['Fz', 'Cz', 'Pz']

        # Pick only the selected channels
        raw_selected = raw.pick_channels(selected_channels)
        return raw_selected


    def prepare_raw_signal(self):
        self.raw_data.pick_types(eeg=True)
        self.raw_data.filter(0.5, 20.5, fir_design='firwin')
        self.raw_data.resample(100)
        return self.raw_data

    def extractBlinksEEG(self, channel):


        # Lets follow the MATLAB code arrangement
        # getBlinkPositions (STEP 1bi)
        self.blinkPositions = getBlinkPosition(self.params,
                                            blinkComp=self.raw_data.get_data(picks=channel)[0], ch=channel)


        # self.blinkPositions = self.blinkPositions[
        #     (self.blinkPositions['endBlinks'] - self.blinkPositions['startBlinks']) / self.sfreq >= 0.05
        #     ]

        # fitBlinks (STEP 1bii)
        fitblinks = FitBlinks(data=self.raw_data.get_data(picks=channel)[0], df=self.blinkPositions,params=self.params)
        fitblinks.dprocess()
        df=fitblinks.frame_blinks


        # getBlinkPositions (STEP 1bi)
        # df = extractBlinks(self.raw_data.get_data(picks=channel)[0],
        #                    self.sfreq, self.params, channel).getBlinksCoordinate()

        # fitBlinks (STEP 1bii)
        # df = FitBlinks(data=self.raw_data.get_data(picks=channel)[0], df=df).frame_blinks
        # In MATLAB, blinkStatProperties is refer as signalData, which is a structure containing the signal data and the blink statistics numberBlinks, numberGoodBlinks, blinkAmpRatio, cutof, bestMedian, and bestRobustStd.
        # This signalData, or blinkStatProperties will be use to reduce the number of candidate signals into a single signal.
        # Later, the step involve
        # 1. Reduce the number of candidate signals based on the blink amp ratios
        # 2. Find the ones that meet the minimum good blink threshold
        # 3. Now see if any candidates meet the good blink ratio criteria
        # 4. Now pick the one with the maximum number of good blinks

        signalData = get_blink_statistic(df, self.params['zThresholds'], signal=self.raw_data.get_data(picks=channel)[0])


        # First reduce on the basis of blink maximum amplitude
        # computeBlinkProperties (Step 2c)
        goodBlinkMask,df=getGoodBlinkMask(df, signalData['bestMedian'], signalData['bestRobustStd'], self.params['zThresholds'] )




        # computeBlinkProperties (Step 2b)
        '''
        Next, for each blink, we will calculate several blink properties including
            - durationBase, durationZero, durationTent, durationHalfBase, durationHalfZero, interBlinkMaxAmp, interBlinkMaxVelBase, interBlinkMaxVelZero, negAmpVelRatioBase, posAmpVelRatioBase, negAmpVelRatioZero, posAmpVelRatioZero, negAmpVelRatioTent, posAmpVelRatioTent, timeShutBase, timeShutZero, timeShutTent, closingTimeZero, reopeningTimeZero, closingTimeTent, reopeningTimeTent, peakTimeBlink, peakTimeTent, and peakMaxTent.
        '''
        df=BlinkProperties(self.raw_data.get_data(picks=channel)[0], df, self.sfreq,self.params).df




        # applyPAVRRestriction (Step 2d)
        # Now apply the final restriction on pAVR to reduce the eye movements. But, in Python, I think we can do this before we calculate the blinkproperties
        # But basiclly, we reduce the number of candidate signals based on the posAmpVelRatioZero (aka pAVR), pAVRThreshold, bestMedian, and bestRobustStd
        # Now apply the final restriction on pAVR to reduce the eye movements
        condition_1 = df['posAmpVelRatioZero'] < self.params['pAVRThreshold']
        condition_2 = df['maxValue'] < (signalData['bestMedian'] - signalData['bestRobustStd'])

        # filters the DataFrame to keep only rows that do not meet both conditions.
        df= df[~(condition_1 & condition_2)]

        # The following line PrepareSelection is not required as I already do this in
        # d = PrepareSelection(signal=self.raw_data.get_data(picks=channel)[0], df=df_filtered, params=self.params, ch=channel).get_param_for_selection()
        self.all_data_info.append(dict(df=df, ch=channel,bproperties=signalData))
        # self.all_data.append(d)

    # @staticmethod
    # def filter_point(ch,all_data_info):
    #     return list(filter(lambda all_data_info: all_data_info['ch'] == ch, all_data_info))[0]
    #

    # def filter_bad_blink(self,df):
    #     # filter_bad = False
    #     if self.filter_bad:
    #         df = df[df['blink_quality'] == 'Good']
    #     return df


    def generate_viz(self,data,df):
        fig_data = [viz_complete_blink_prop(data, row, self.sfreq) for index, row in df.iterrows()]

        return fig_data
    def get_blink_stat(self):
        for channel in self.channel_list:
            self.extractBlinksEEG(channel)
        all_data_info = self.all_data_info

        dataframes = []
        for entry in all_data_info:
            # Flatten the `bproperties` dictionary and add `ch` as a column
            row = entry['bproperties'].copy()
            row['ch'] = entry['ch']

            # Append as a new row to the dataframe list
            dataframes.append(row)
        df_stat = pd.DataFrame(dataframes)

        params = self.params


        # In MATLAB, this is refer as step 1biii
        df_stat = filter_blink_amplitude_ratios(df_stat, params)

        # Find the ones that meet the minimum good blink threshold
        df_stat = filter_good_blinks(df_stat, params)

        # Now see if any candidates meet the good blink ratio criteria
        df_stat = filter_good_ratio(df_stat, params)

        # Now pick the one with the maximum number of good blinks
        df_stat = select_max_good_blinks(df_stat)
        ch= df_stat.loc[df_stat['select'], 'ch'].tolist()[0]
        selected_indices = df_stat.loc[df_stat['select']].index.tolist()[0]
        df=all_data_info[selected_indices]['df']
        df.to_excel('blink_stat_002.xlsx')
        logging.warning('THIS SECTION YET TO BE IMPLEMENTED OR REVISED')
        # ch_blink_stat = pd.DataFrame(self.all_data)
        # ch_selected = ChannelSelection(df=ch_blink_stat, params=self.params).df
        # ch = ch_selected.loc[0, 'ch']
        # nGoodBlinks = ch_selected.loc[0, 'numberGoodBlinks']
        nGoodBlinks=100
        data = self.raw_data.get_data(picks=ch)[0]


        annot_description = self.annot_label if self.annot_label else 'eye_blink'
        annot = create_annotation(df, self.sfreq, annot_description)
        if self.viz_data:
            fig_data=self.generate_viz(data,df)
            return annot, ch, nGoodBlinks, fig_data, df

        return annot, ch, nGoodBlinks,df
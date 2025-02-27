# """
# Compact script for development purpose,
# From this simple script,we can identified where to improve the code
# This scrip take advantage the processing power of pandas-apply. and much more compact
#
# """
# import logging
# import os
#
# import hickle as hkl
# import matplotlib
# import mne
# import pandas as pd
#
# from eeg_blinks import default_setting
# from eeg_blinks.utilities.getBlinkPositions_vislab import get_blink_position
# from eeg_blinks.utilities.misc import mad_matlab
# from eeg_blinks.utilities.zero_crossing import *
# from eeg_blinks.utilities.zero_crossing import _maxPosVelFrame, _get_left_base, _get_right_base, _get_half_height
#
# logging.getLogger().setLevel(logging.INFO)
#
#
# def fitBlinks(candidateSignal=None, blinkPositions=None):
#     # filename = 'data_to_fitBlinks.hkl'
#     # hkl.dump([candidateSignal, blinkPositions], filename)
#
#     startBlinks = blinkPositions['start_blink']
#     endBlinks = blinkPositions['end_blink']
#
#     maxValues, maxFrames = zip(*[_get_max_frame(candidateSignal, dstartBlinks, dendBlinks) for
#                                  dstartBlinks, dendBlinks in zip(startBlinks, endBlinks)])
#     ## Calculate the fits
#
#
#     maxFrames = np.array(maxFrames)
#     maxValues = np.array(maxValues)
#     outerStarts = np.append(0, maxFrames[0:-1])
#     outerEnds = np.append(maxFrames[1:], candidateSignal.size)
#
#     df = pd.DataFrame(dict(maxFrames=maxFrames, maxValues=maxValues, startBlinks=startBlinks, endBlinks=endBlinks,
#                            outerStarts=outerStarts, outerEnds=outerEnds))
#
#     blinkVelocity = np.diff(candidateSignal, axis=0)
#
#     import hickle as hkl
#     filename = 'data_to_get_zero_crossing_pd.hkl'
#     hkl.dump([candidateSignal, blinkVelocity, df], filename)
#
#
# def fit_blink_pd_approach():
#     filename = 'data_to_get_zero_crossing_pd.hkl'
#     # hkl.dump([candidateSignal, blinkVelocity,df], filename)
#
#     data, blinkVelocity, df = hkl.load(filename)
#
#     baseFraction = 0.1  # Fraction from top and bottom
#     df[['leftZero', 'rightZero']] = df.apply(lambda x: left_right_zero_crossing(data, x['maxFrames'], x['outerStarts'],
#                                                                                 x['outerEnds']), axis=1,
#                                              result_type="expand")
#
#     df[['maxPosVelFrame', 'maxNegVelFrame']] = df.apply(
#         lambda x: _maxPosVelFrame(blinkVelocity, x['maxFrames'], x['leftZero'],
#                                   x['rightZero']), axis=1, result_type="expand")
#
#     ## Lets check some condition especially for data with anamoly
#
#     df = df[df['outerStarts'] < df['maxPosVelFrame']]  # Filter and take only row that normal
#     df['leftBase'] = df.apply(lambda x: _get_left_base(blinkVelocity, x['outerStarts'], x['maxPosVelFrame']), axis=1)
#
#     df['rightBase'] = df.apply(lambda x: _get_right_base(data, blinkVelocity, x['outerEnds'], x['maxNegVelFrame']),
#                                axis=1)
#
#     cols_half_height = ['leftZeroHalfHeight', 'rightZeroHalfHeight', 'leftBaseHalfHeight', 'rightBaseHalfHeight']
#     df[cols_half_height] = df.apply(lambda x: _get_half_height(data, x['maxFrames'], x['leftZero'], x['rightZero'],
#                                                                x['leftBase'], x['outerEnds']), axis=1,
#                                     result_type="expand")
#
#     cols_fit_range = ['xLeft', 'xRight', 'leftRange', 'rightRange',
#                       'blinkBottomPoint_l_Y', 'blinkBottomPoint_l_X', 'blinkTopPoint_l_Y', 'blinkTopPoint_l_X',
#                       'blinkBottomPoint_r_X', 'blinkBottomPoint_r_Y', 'blinkTopPoint_r_X', 'blinkTopPoint_r_Y']
#
#     df[cols_fit_range] = df.apply(lambda x: compute_fit_range(data, x['maxFrames'], x['leftZero'], x['rightZero'],
#                                                               baseFraction, top_bottom=True), axis=1,
#                                   result_type="expand")
#
#     df = df.dropna()
#     df.reset_index(drop=True)
#
#     cols_lines_intesection = ['leftSlope', 'rightSlope', 'averLeftVelocity', 'averRightVelocity',
#                               'rightR2', 'leftR2', 'xIntersect', 'yIntersect', 'leftXIntercept',
#                               'rightXIntercept', 'xLineCross_l', 'yLineCross_l', 'xLineCross_r', 'yLineCross_r']
#
#     df[cols_lines_intesection] = df.apply(lambda x: lines_intersection(xRight=x['xRight'], xLeft=x['xLeft'],
#                                                                        yRight=data[x['xRight']], yLeft=data[x['xLeft']],
#                                                                        dic_type=False), axis=1, result_type="expand")
#
#     print(df)
#     filename = 'data_to_get_zero_crossing_pd_viz.hkl'
#     hkl.dump([data, df], filename)
#
# def _goodblink_based_corr_median_std(df,correlationThreshold):
#     R2 = df['leftR2'] >= correlationThreshold
#     R3 = df['rightR2'] >= correlationThreshold
#
#     # Now calculate the cutoff ratios -- use default for the values
#     # df['R2']=R2
#     # df['R3']=R3
#     # df['res']=R2.values & R3.values
#     good_data=df.loc[R2.values & R3.values,:]
#     bestValues=good_data['maxValues'].array
#
#     specified_median = np.nanmedian(bestValues)
#     specified_std = 1.4826 * mad_matlab(bestValues)
#
#     return R2, R3,specified_median,specified_std
#
# def get_mask(df, indicesNaN,  correlationThreshold, zScoreThreshold):
#     """
#     The calculation of bestmedian,worst median, worrst rbobustst
#     is from https://github.com/VisLab/EEG-Blinks/blob/16b6ea04101ecfa74fb1c9cbceb037324572687e/blinker/utilities/extractBlinks.m#L97
#
#     :param df:
#     :param indicesNaN:
#     :param correlationThreshold:
#     :param zScoreThreshold:
#     :return:
#     """
#     R1 = ~indicesNaN
#
#     R2, R3,specified_median,specified_std=_goodblink_based_corr_median_std(df,correlationThreshold)
#     # Filter
#     R4 = df['maxValues'] >= max(0, specified_median - zScoreThreshold * specified_std)
#     R5 = df['maxValues'] <= specified_median + zScoreThreshold * specified_std
#     df1 = pd.concat([R1, R2, R3, R4, R5], axis=1)
#     return df1,specified_median,specified_std
#
#
# def get_good_blink_mask(df, z_thresholds):
#
#
#
#     # specified_median, specified_std = 94.156208105121580, 28.368641236303578  # Need to find where is the source
#
#     # warnings.warn('I am not so sure from where we get this value')
#
#
#
#     ## These is the default value
#     correlationThreshold_s1,correlationThreshold_s2,zScoreThreshold_s1,zScoreThreshold_s2 = z_thresholds
#
#     # df = pd.DataFrame(blinkFits)
#     df['rightR2'] = df['rightR2'].abs()
#     df_data = df[['leftR2', 'rightR2', 'maxValues']]
#
#     indicesNaN = df_data.isnull().any(1)
#
#     ### Get mask
#
#     goodMaskTop,bestMedian,bestRobustStd = get_mask(df_data, indicesNaN, correlationThreshold_s1, zScoreThreshold_s1)
#     goodBlinkMask_s1 = goodMaskTop.all(1)
#
#     df_s2,worstMedian,worstRobustStd = get_mask(df_data, indicesNaN,  correlationThreshold_s2, zScoreThreshold_s2)
#
#     ff_s2 = df_s2.all(1)
#     df_s2a = pd.concat([goodBlinkMask_s1, ff_s2], axis=1)
#     goodBlinkMask = df_s2a.any(1)
#     df['blink_quality'] = 'Good'
#     df[['blink_quality']] = df[['blink_quality']].where(goodBlinkMask, other='Reject')
#     return df,bestMedian,bestRobustStd
#
# def _get_param_for_Selection():
#     filename = 'data_to_get_zero_crossing_pd_viz.hkl'
#     ch='mock_channel'
#     signal, df = hkl.load(filename)
#     df['rightR2'] = df['rightR2'].abs()
#     params=default_setting.params
#     blinkMask = np.zeros(signal.size, dtype=bool)
#
#     for leftZero_y, rightZero_x in zip(df['leftZero'] - 1, df['rightZero'], ):
#         blinkMask[leftZero_y:rightZero_x] = True
#
#     outsideBlink = np.logical_and(signal > 0, ~blinkMask)
#
#     insideBlink = np.logical_and(signal > 0, blinkMask)
#
#     blinkAmpRatio = np.mean(signal[insideBlink]) / np.mean(signal[outsideBlink])  # 2.0629878
#
#     # Now calculate the cutoff ratios -- use default for the values
#     R2_top,R3_top,bestMedian,bestRobustStd=_goodblink_based_corr_median_std(df,params['correlationThresholdTop'])
#     R2_bot,R3_bot,worstMedian,worstRobustStd=_goodblink_based_corr_median_std(df,params['correlationThresholdBottom'])
#
#
#
#     true_top=R2_top.values & R3_top.values
#     nTRue_top=true_top.sum()
#     bestValues=df.loc[true_top,'maxValues']
#     true_bot=R2_bot.values & R3_bot.values
#     nTRue=true_bot.sum()
#     worstValues = df.loc[~true_bot,'maxValues']
#     goodValues =df.loc[true_bot,'maxValues']
#
#     allValues = df['maxValues']
#
#
#     cutoff = (bestMedian * worstRobustStd + worstMedian * bestRobustStd) / (
#             bestRobustStd + worstRobustStd)  # 10.8597297664580
#
#     all_X = np.sum(np.logical_and(allValues <= bestMedian + 2 * bestRobustStd,
#                                   allValues >= bestMedian - 2 * bestRobustStd))  # 162
#
#     if all_X != 0:
#         goodRatio = np.sum(np.logical_and(goodValues <= bestMedian + 2 * bestRobustStd,
#                                           goodValues >= bestMedian - 2 * bestRobustStd)) / all_X  # 0.253086419753086
#     else:
#         goodRatio = np.nan
#         # numberGoodBlinks = np.sum ( goodMaskBottom )  # 50
#     # numberGoodBlinks = np.sum(goodMaskBottom)
#     numberGoodBlinks=true_bot.sum()
#     ALL_DATA = [ch, blinkAmpRatio, bestMedian, bestRobustStd, cutoff, goodRatio, numberGoodBlinks,df]
#     header_eb_label = ['ch', 'blinkAmpRatio', 'bestMedian', 'bestRobustStd', 'cutoff', 'goodRatio',
#                        'numberGoodBlinks']
#     data_blink = dict(zip(header_eb_label, ALL_DATA))
#
#
# def extracBlinkProperties():
#     """
#     Return a structure with blink shapes and properties for individual blinks
#
#     :return:
#     """
#
#     srate = 100
#
#
#     z_thresholds= (0.90, 0.98, 2,5)  # also availabe at getGOodBlinkMask: Orignally frame like this (0.90, 2, 0.98, 5)
#
#
#
#     filename = 'data_to_get_zero_crossing_pd_viz.hkl'
#
#     data, df = hkl.load(filename)
#     df.reset_index(drop=True,
#                    inplace=True)  # To ensure all index are reset, since we concat some index along the pipeline later
#
#     # # SLice only good blink. Expect lesser selection number
#     df,bestMedian,bestRobustStd = get_good_blink_mask(df, z_thresholds)
#
#     signal_l = data.shape[0]
#     blinkVelocity = np.diff(data)  # to cross check whether this is correct?
#     cols_int = ['rightBase']
#     df[cols_int] = df[cols_int].astype(int)
#
#     ## Blink durations
#     df['durationBase'] = (df['rightBase'] - df['leftBase']) / srate
#     df['durationTent'] = (df['rightXIntercept'] - df['leftXIntercept']) / srate
#     df['durationZero'] = (df['rightZero'] - df['leftZero']) / srate
#     df['durationHalfBase'] = (df['rightBaseHalfHeight'] - df['leftBaseHalfHeight'] + 1) / srate
#     df['durationHalfZero'] = (df['rightZeroHalfHeight'] - df['leftZeroHalfHeight'] + 1) / srate
#
#     ## Blink amplitude-velocity ratio from zero to max
#     df['peaksPosVelZero'] = df.apply(
#         lambda x: x['leftZero'] + np.argmax(blinkVelocity[x['leftZero']:x['maxFrames'] + 1]), axis=1)
#
#     # TODO TO remove the minus 1 >> df['maxFrames']-1
#     df['RRC'] = data[df['maxFrames'] - 1] / blinkVelocity[df['peaksPosVelZero']]
#     df['posAmpVelRatioZero'] = (100 * abs(df['RRC'])) / srate
#
#     df['downStrokevelFrame_del'] = df.apply(
#         lambda x: x['maxFrames'] + np.argmin(blinkVelocity[x['maxFrames']:x['rightZero'] + 1]), axis=1)
#
#     df['TTT'] = data[df['maxFrames'] - 1] / blinkVelocity[df['downStrokevelFrame_del']]
#     df['negAmpVelRatioZero'] = (100 * abs(df['TTT'])) / srate
#
#     ## Blink amplitude-velocity ratio from base to max
#
#     df['peaksPosVelBase'] = df.apply(
#         lambda x: x['leftBase'] + np.argmax(blinkVelocity[x['leftBase']:x['maxFrames'] + 1]), axis=1)
#     df['KKK'] = data[df['maxFrames'] - 1] / blinkVelocity[df['peaksPosVelBase']]
#     df['posAmpVelRatioBase'] = (100 * abs(df['KKK'])) / srate
#
#     df['downStroke_del'] = df.apply(
#         lambda x: x['maxFrames'] + np.argmin(blinkVelocity[x['maxFrames']:x['rightBase'] + 1]), axis=1)
#     df['KKK'] = data[df['maxFrames'] - 1] / blinkVelocity[df['downStroke_del']]
#     df['negAmpVelRatioBase'] = (100 * abs(df['KKK'])) / srate
#
#     ## Blink amplitude-velocity ratio estimated from tent slope
#
#     # TODO TO remove the minus 1 >> df['maxFrames']-1
#     df['pop'] = data[df['maxFrames'] - 1] / df['averRightVelocity']
#     df['negAmpVelRatioTent'] = (100 * abs(df['pop'])) / srate
#
#     df['opi'] = data[df['maxFrames'] - 1] / df['averLeftVelocity']
#     df['WE'] = (100 * abs(df['opi']))
#     df['posAmpVelRatioTent'] = df['WE'] / srate
#
#     ## Time zero shut
#     shutAmpFraction = 0.9
#     df['closingTimeZero'] = (df['maxFrames'] - df['leftZero']) / srate
#
#     df['reopeningTimeZero'] = (df['rightZero'] - df['maxFrames']) / srate
#
#     df['ampThreshhold'] = shutAmpFraction * df['maxValues']
#     df['start_shut_tzs'] = df.apply(
#         lambda x: np.argmax(data[x['leftZero']:x['rightZero'] + 1] >= x['ampThreshhold']), axis=1)
#
#     df['endShut_tzs'] = df.apply(
#         lambda x: np.argmax(data[x['leftZero']:x['rightZero'] + 1][x['start_shut_tzs'] + 1:-1] <
#                             shutAmpFraction * x['maxValues']), axis=1)
#
#
#
#     ## PLease expect error here, some value maybe zero or lead to empty cell
#     df['endShut_tzs'] = df['endShut_tzs'] + 1  ## temporary  to delete
#     df['timeShutZero'] = df.apply(
#         lambda x: 0 if x['endShut_tzs'] == np.isnan else x['endShut_tzs'] / srate, axis=1)
#
#     ## Time base shut
#     shutAmpFraction = 0.9
#     df['ampThreshhold_tbs'] = shutAmpFraction * df['maxValues']
#     df['start_shut_tbs'] = df.apply(
#         lambda x: np.argmax(data[x['leftBase']:x['rightBase'] + 1] >= x['ampThreshhold_tbs']), axis=1)
#
#     df['endShut_tbs'] = df.apply(
#         lambda x: np.argmax(data[x['leftBase']:x['rightBase'] + 1][x['start_shut_tbs']:-1] <
#                             shutAmpFraction * x['maxValues']), axis=1)
#
#
#     df['timeShutBase'] = df.apply(
#         lambda x: 0 if x['endShut_tbs'] == np.isnan else (x['endShut_tbs'] / srate), axis=1)
#
#     ## Time shut tent
#     df['closingTimeTent'] = (df['xIntersect'] - df['leftXIntercept']) / srate
#     df['reopeningTimeTent'] = (df['rightXIntercept'] - df['xIntersect']) / srate
#
#     df['ampThreshhold_tst'] = shutAmpFraction * df['maxValues']
#
#     df[['leftXIntercept_int', 'rightXIntercept_int']] = df[['leftXIntercept', 'rightXIntercept']].astype(
#         int)
#     df['start_shut_tst'] = df.apply(
#         lambda x: np.argmax(data[x['leftXIntercept_int']:x['rightXIntercept_int'] + 1] >= x['ampThreshhold']),
#         axis=1)
#
#     df['endShut_tst'] = df.apply(
#         lambda x: np.argmax(data[x['leftXIntercept_int']:x['rightXIntercept_int'] + 1][x['start_shut_tst']:-1] <
#                             shutAmpFraction * x['maxValues']), axis=1)
#
#
#     df['timeShutTent'] = df.apply(
#         lambda x: 0 if x['endShut_tst'] == np.isnan else (x['endShut_tst'] / srate), axis=1)
#
#     ## Other times
#     df['peakMaxBlink '] = df['maxValues']
#     df['peakMaxTent'] = df['yIntersect']
#     df['peakTimeTent'] = df['xIntersect'] / srate
#     df['peakTimeBlink'] = df['maxFrames'] / srate
#
#     dfcal = df[['maxFrames', 'peaksPosVelBase', 'peaksPosVelZero']]
#
#     df_t = pd.DataFrame.from_records([[signal_l] * 3], columns=['maxFrames', 'peaksPosVelBase', 'peaksPosVelZero'])
#
#     dfcal = pd.concat([dfcal, df_t]).reset_index(drop=True)
#
#     dfcal['ibmx'] = dfcal.maxFrames.diff().shift(-1)
#
#     dfcal['interBlinkMaxAmp'] = dfcal['ibmx'] / srate
#
#     dfcal['ibmvb'] = 1 - dfcal['peaksPosVelBase']
#     dfcal['interBlinkMaxVelBase'] = dfcal['ibmvb'] / srate  # peaksPosVelBase == velFrame
#
#     dfcal['ibmvz'] = 1 - dfcal['peaksPosVelZero']
#     dfcal['interBlinkMaxVelZero'] = dfcal['ibmvz'] / srate
#
#     dfcal.drop(dfcal.tail(1).index, inplace=True)  # drop last n rows# peaksPosVelZero == velFrame
#     dfnew = df[['maxValues', 'posAmpVelRatioZero']]
#
#     pAVRThreshold = 3
#     R1 = dfnew['posAmpVelRatioZero'] < pAVRThreshold
#
#     th_bm_brs = bestMedian - bestRobustStd
#     R2 = dfnew['maxValues'] < th_bm_brs
#     pMask = pd.concat([R1, R2], axis=1)
#     pMasks = pMask.all(1)
#     df_res = pd.merge(df, dfcal, on=['maxFrames'])
#     # df_res = pd.concat([df, dfcal], axis=1)
#     df_res = df_res[~pMasks].reset_index(drop=True)
#
#     print(df)
#     filename = 'data_to_selected_blink_pd_viz.hkl'
#     hkl.dump([data, df_res], filename)
#
#
#
#
# def viz_fit_blink_pd_approach():
#     from eeg_blinks.viz.viz_pd import viz_complete_blink_prop
#     import mne
#
#     title = 'sddd'
#     rep = mne.Report(title=title)
#
#     filename = 'data_to_selected_blink_pd_viz.hkl'
#
#     data, df = hkl.load(filename)
#     print(df.dtypes)
#     cols_int = ['rightBase']
#     df[cols_int] = df[cols_int].astype(int)
#     df.to_excel('da.xlsx')
#     fig_good_blink = []
#     fig_bad_blink = []
#     # for index, row in df.iloc[1:].iterrows():
#     for index, row in df.iterrows():
#
#         dfig = viz_complete_blink_prop(data, row)
#
#         if row['blink_quality'] == 'Good':
#             fig_good_blink.append(dfig)
#         else:
#             fig_bad_blink.append(dfig)
#
#     all_cap_good = ['Good'] * len(fig_good_blink)
#     for disfig, discaption in zip(fig_good_blink, all_cap_good):
#         # lcaption=discaption
#         rep.add_figs_to_section(disfig, captions=discaption, section='Good blink')
#
#     all_cap_bad = ['Good'] * len(fig_bad_blink)
#     for disfig, discaption in zip(fig_bad_blink, all_cap_bad):
#         # lcaption=discaption
#         rep.add_figs_to_section(disfig, captions=discaption, section='bad blink')
#
#     spath = 'dreport.html'
#     rep.save(spath, overwrite=True, open_browser=False)
#
#
# def open_file():
#     matplotlib.use('TkAgg')
#     sample_data_folder = mne.datasets.sample.data_path()
#     sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                         'sample_audvis_filt-0-40_raw.fif')
#     raw = mne.io.read_raw_fif(sample_data_raw_file)
#
#     raw.pick_types(eeg=True)
#     raw = raw.load_data()
#     ch_list = raw.ch_names
#     raw.filter(0.5, 20.5, fir_design='firwin')
#     raw.resample(100)
#     # raw.save('raw_audvis_resampled.fif')
#     # drange=[f'EEG 00{X}' for X in range (10)]
#     drange = ['EEG 002']
#     to_drop_ch = list(set(raw.ch_names) - set(drange))
#     raw = raw.drop_channels(to_drop_ch)
#
#     params = default_setting.params
#     # annot_description = kwargs.get('annot_label', 'eye_blink')
#
#     # ch_list = raw.ch_names if include is None else include
#     sfreq = raw.info['sfreq']
#
#     logging.info('Get the blink position. This may take some time since channel is process at a time ')
#     ch_list = raw.ch_names
#     blinkPositions_list = [get_blink_position(params, sfreq, blink_component=raw.get_data(picks=ch)[0], ch=ch) for ch in
#                            ch_list]
#
#     return blinkPositions_list, raw, params
#
#
# def _get_max_frame(candidateSignal, startBlinks, endBlinks):
#     blinkRange = np.arange(startBlinks, endBlinks)
#     dff = candidateSignal[startBlinks:endBlinks]
#
#     maxValues = np.amax(dff)
#     maxFrames = blinkRange[np.argmax(dff)]
#
#     return maxValues, maxFrames
#
#
# def prepare_df_compatible(blinkPositions_list, raw, params):
#     # signalData = [_extracBlinks(ch_data, raw, params) for ch_data in blinkPositions_list]
#     srate=100
#     ch_data = blinkPositions_list[0]
#     ch = ch_data['ch']
#     print(ch)
#
#     signal_eeg = raw.get_data(picks=ch)[0]
#
#     # filename = 'data_to_fitBlinks.hkl'
#     # hkl.dump([candidateSignal, blinkPositions], filename)
#
#     startBlinks = ch_data['start_blink']  # ch_data is equivalent to blinkPosition
#     endBlinks = ch_data['end_blink']
#
#     maxValues, maxFrames = zip(*[_get_max_frame(signal_eeg, dstartBlinks, dendBlinks) for
#                                  dstartBlinks, dendBlinks in zip(startBlinks, endBlinks)])
#     ## Calculate the fits
#
#     baseFraction = 0.1  # Fraction from top and bottom
#     maxFrames = np.array(maxFrames)
#     maxValues = np.array(maxValues)
#     outerStarts = np.append(0, maxFrames[0:-1])
#     outerEnds = np.append(maxFrames[1:], signal_eeg.size)
#
#     df = pd.DataFrame(dict(maxFrames=maxFrames, maxValues=maxValues, startBlinks=startBlinks, endBlinks=endBlinks,
#                            outerStarts=outerStarts, outerEnds=outerEnds))
#
#     ## TODO remove blinks that arent separated
#
#     df['blink_duration']=(df['endBlinks']-df['startBlinks'])/srate
#
#     df = df[df.blink_duration.ge(0.05) ].reset_index(drop=True)
#     # dfs=df.drop(df['blink_duration']>0.05].index)
#     blinkVelocity = np.diff(signal_eeg, axis=0)
#
#     # Data are all store within the same folder of this main file
#     import hickle as hkl
#     filename = 'data_to_get_zero_crossing_pd.hkl'
#     hkl.dump([signal_eeg, blinkVelocity, df], filename)
#
#
# blinkPositions_list,raw,params=open_file() # Lets load the file and find the start end of the blink
# # #
# prepare_df_compatible(blinkPositions_list,raw,params) # Since the objective is to work with pandas, transform all info to pd compatible
#
# # fit_blink_pd_approach() # Working and tally with viz check
# # _get_param_for_Selection() # WIP
# extracBlinkProperties() # Pending, dont expect any result from this
# # viz_fit_blink_pd_approach()  # This is working
# # h = 1

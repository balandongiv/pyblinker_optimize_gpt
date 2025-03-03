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
# import numpy as np
# import pandas as pd
#
# from eeg_blinks import default_setting
#
# from eeg_blinks.utilities.getBlinkPositions_vislab import get_blink_position
# from eeg_blinks.utilities.misc import mad_matlab
# from eeg_blinks.utilities.zero_crossing import *
# from eeg_blinks.utilities.zero_crossing import _max_pos_vel_frame, _get_left_base, _get_right_base, _get_half_height
#
# logging.getLogger().setLevel(logging.INFO)
#
# matplotlib.use('TkAgg')
#
# hh = 1
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
# def fit_blink_pd_approach(candidate_signal=None, df=None):
#     if candidate_signal is None:
#         filename = 'data_to_get_zero_crossing_pd_MFF.hkl'
#         # hkl.dump([candidateSignal, blinkVelocity,df], filename)
#         candidate_signal, df = hkl.load(filename)
#
#     blinkVelocity = np.diff(candidate_signal, axis=0)
#     baseFraction = 0.1  # Fraction from top and bottom
#     df[['leftZero', 'rightZero']] = df.apply(lambda x: left_right_zero_crossing(candidate_signal, x['maxFrames'], x['outerStarts'],
#                                                                                 x['outerEnds']), axis=1,
#                                              result_type="expand")
#
#     df[['maxPosVelFrame', 'maxNegVelFrame']] = df.apply(
#         lambda x: _max_pos_vel_frame(blinkVelocity, x['maxFrames'], x['leftZero'],
#                                   x['rightZero']), axis=1, result_type="expand")
#
#     ## Lets check some condition especially for candidate_signal with anamoly
#
#     df = df[df['outerStarts'] < df['maxPosVelFrame']]  # Filter and take only row that normal
#     df['leftBase'] = df.apply(lambda x: _get_left_base(blinkVelocity, x['outerStarts'], x['maxPosVelFrame']), axis=1)
#
#     df = df.dropna()
#     # TODO
#
#     df['rightBase'] = df.apply(lambda x: _get_right_base(candidate_signal, blinkVelocity, x['outerEnds'], x['maxNegVelFrame']),
#                                axis=1)
#
#     cols_half_height = ['leftZeroHalfHeight', 'rightZeroHalfHeight', 'leftBaseHalfHeight', 'rightBaseHalfHeight']
#     df[cols_half_height] = df.apply(lambda x: _get_half_height(candidate_signal, x['maxFrames'], x['leftZero'], x['rightZero'],
#                                                                x['leftBase'], x['outerEnds']), axis=1,
#                                     result_type="expand")
#
#     cols_fit_range = ['xLeft', 'xRight', 'leftRange', 'rightRange',
#                       'blinkBottomPoint_l_Y', 'blinkBottomPoint_l_X', 'blinkTopPoint_l_Y', 'blinkTopPoint_l_X',
#                       'blinkBottomPoint_r_X', 'blinkBottomPoint_r_Y', 'blinkTopPoint_r_X', 'blinkTopPoint_r_Y']
#
#     df[cols_fit_range] = df.apply(lambda x: compute_fit_range(candidate_signal, x['maxFrames'], x['leftZero'], x['rightZero'],
#                                                               baseFraction, top_bottom=True), axis=1,
#                                   result_type="expand")
#
#     df = df.dropna()
#
#     df['nsize_xLeft'] = df.apply(lambda x: x['xLeft'].size, axis=1)
#     df['nsize_xRight'] = df.apply(lambda x: x['xRight'].size, axis=1)
#
#     df = df[~(df['nsize_xLeft'] <= 1) & ~(df['nsize_xRight'] <= 1)]
#
#     df.reset_index(drop=True, inplace=True)
#     cols_lines_intesection = ['leftSlope', 'rightSlope', 'averLeftVelocity', 'averRightVelocity',
#                               'rightR2', 'leftR2', 'xIntersect', 'yIntersect', 'leftXIntercept',
#                               'rightXIntercept', 'xLineCross_l', 'yLineCross_l', 'xLineCross_r', 'yLineCross_r']
#
#     df[cols_lines_intesection] = df.apply(lambda x: lines_intersection(xRight=x['xRight'], xLeft=x['xLeft'],
#                                                                        yRight=candidate_signal[x['xRight']], yLeft=candidate_signal[x['xLeft']],
#                                                                        dic_type=False), axis=1, result_type="expand")
#
#     # print(df)
#     if candidate_signal is None:
#         filename = 'data_to_get_zero_crossing_pd_viz_MFF.hkl'
#         hkl.dump([candidate_signal, df], filename)
#     else:
#         return df
#
#
# def _goodblink_based_corr_median_std(df, correlationThreshold):
#     R2 = df['leftR2'] >= correlationThreshold
#     R3 = df['rightR2'] >= correlationThreshold
#
#     # Now calculate the cutoff ratios -- use default for the values
#     good_data = df.loc[R2.values & R3.values, :]
#     bestValues = good_data['maxValues'].array
#
#     specified_median = np.nanmedian(bestValues)
#     specified_std = 1.4826 * mad_matlab(bestValues)
#
#     return R2, R3, specified_median, specified_std
#
#
# def get_mask_optimise(df, indicesNaN, correlationThreshold, zScoreThreshold):
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
#     R2, R3, specified_median, specified_std = _goodblink_based_corr_median_std(df, correlationThreshold)
#
#     R4 = df['maxValues'] >= max(0, specified_median - zScoreThreshold * specified_std)
#     R5 = df['maxValues'] <= specified_median + zScoreThreshold * specified_std
#     bool_test = R1.values & R2.values & R3.values & R4.values & R5.values
#
#     return bool_test, specified_median, specified_std
#
#
# def get_good_blink_mask(df, z_thresholds):
#     # specified_median, specified_std = 94.156208105121580, 28.368641236303578  # Need to find where is the source
#
#     # warnings.warn('I am not so sure from where we get this value')
#
#     ## These is the default value
#     correlationThreshold_s1, correlationThreshold_s2, zScoreThreshold_s1, zScoreThreshold_s2 = z_thresholds
#
#     df['rightR2'] = df['rightR2'].abs()
#     df_data = df[['leftR2', 'rightR2', 'maxValues']]
#
#     indicesNaN = df_data.isnull().any(1)
#
#     ### GET MASK OPTIMISE
#
#     goodMaskTop_bool, bestMedian, bestRobustStd = get_mask_optimise(df_data, indicesNaN, correlationThreshold_s1,
#                                                                     zScoreThreshold_s1)
#
#     df_s2_bool, worstMedian, worstRobustStd = get_mask_optimise(df_data, indicesNaN, correlationThreshold_s2,
#                                                                 zScoreThreshold_s2)
#
#     goodBlinkMask = np.reshape(goodMaskTop_bool | df_s2_bool, (-1, 1))  # Get any TRUE
#     df['blink_quality'] = 'Good'
#     df[['blink_quality']] = df[['blink_quality']].where(goodBlinkMask, other='Reject')
#     return df, bestMedian, bestRobustStd
#
#
# def _get_param_for_selection(signal=None, df=None, params=None, ch=None):
#     if signal is None:
#         filename = 'data_to_get_zero_crossing_pd_viz_MFF.hkl'
#         ch = 'mock_channel'
#         signal, df = hkl.load(filename)
#         params = default_setting.params
#
#     df['rightR2'] = df['rightR2'].abs()
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
#     R2_top, R3_top, bestMedian, bestRobustStd = _goodblink_based_corr_median_std(df, params['correlationThresholdTop'])
#     R2_bot, R3_bot, worstMedian, worstRobustStd = _goodblink_based_corr_median_std(df,
#                                                                                    params['correlationThresholdBottom'])
#
#     true_top = R2_top.values & R3_top.values
#     nTRue_top = true_top.sum()
#     bestValues = df.loc[true_top, 'maxValues']
#     true_bot = R2_bot.values & R3_bot.values
#     nTRue = true_bot.sum()
#     worstValues = df.loc[~true_bot, 'maxValues']
#     goodValues = df.loc[true_bot, 'maxValues']
#
#     allValues = df['maxValues']
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
#     numberGoodBlinks = true_bot.sum().item()
#     all_data = [ch, blinkAmpRatio, bestMedian, bestRobustStd, cutoff, goodRatio, numberGoodBlinks, df]
#     header_eb_label = ['ch', 'blinkAmpRatio', 'bestMedian', 'bestRobustStd', 'cutoff', 'goodRatio',
#                        'numberGoodBlinks']
#     data_blink = dict(zip(header_eb_label, all_data))
#
#     return data_blink
#
#
# def _start_shut(arr, s, e, th):
#     """
#
#     df['start_shut_tst'] = df.apply(
#     lambda x: np.argmax(candidate_signal[x['leftXIntercept_int']:x['rightXIntercept_int'] + 1] >= x['ampThreshhold']),
#
#     While it very tempting to df = df.astype({"start_shut_tst": int}), but since we deal with nan, this is imposible
#     or
#
#     np.argmax(arr[s:e + 1] >= th).astype(int)
#     axis=1)
#     :param arr:
#     :param s:
#     :param e:
#     :param th:
#     :return:
#     """
#     try:
#
#         return np.argmax(arr[s:e + 1] >= th)
#     except ValueError:
#         return np.nan
#
#
# def extracBlinkProperties(srate):
#     """
#     Return a structure with blink shapes and properties for individual blinks
#
#     :return:
#     """
#
#     # srate = 100
#     shutAmpFraction = 0.9
#     z_thresholds = (0.90, 0.98, 2, 5)  # also availabe at getGOodBlinkMask: Orignally frame like this (0.90, 2, 0.98, 5)
#
#     filename = 'data_to_get_zero_crossing_pd_viz_MFF.hkl'
#
#     candidate_signal, df = hkl.load(filename)
#     df.reset_index(drop=True,
#                    inplace=True)  # To ensure all index are reset, since we concat some index along the pipeline later
#
#     # # SLice only good blink. Expect lesser selection number
#     df, bestMedian, bestRobustStd = get_good_blink_mask(df, z_thresholds)
#
#     signal_l = candidate_signal.shape[0]
#     blinkVelocity = np.diff(candidate_signal)  # to cross check whether this is correct?
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
#     df['RRC'] = candidate_signal[df['maxFrames'] - 1] / blinkVelocity[df['peaksPosVelZero']]
#     df['posAmpVelRatioZero'] = (100 * abs(df['RRC'])) / srate
#
#     df['downStrokevelFrame_del'] = df.apply(
#         lambda x: x['maxFrames'] + np.argmin(blinkVelocity[x['maxFrames']:x['rightZero'] + 1]), axis=1)
#
#     df['TTT'] = candidate_signal[df['maxFrames'] - 1] / blinkVelocity[df['downStrokevelFrame_del']]
#     df['negAmpVelRatioZero'] = (100 * abs(df['TTT'])) / srate
#
#     ## Blink amplitude-velocity ratio from base to max
#
#     df['peaksPosVelBase'] = df.apply(
#         lambda x: x['leftBase'] + np.argmax(blinkVelocity[x['leftBase']:x['maxFrames'] + 1]), axis=1)
#     df['KKK'] = candidate_signal[df['maxFrames'] - 1] / blinkVelocity[df['peaksPosVelBase']]
#     df['posAmpVelRatioBase'] = (100 * abs(df['KKK'])) / srate
#
#     df['downStroke_del'] = df.apply(
#         lambda x: x['maxFrames'] + np.argmin(blinkVelocity[x['maxFrames']:x['rightBase'] + 1]), axis=1)
#     df['KKK'] = candidate_signal[df['maxFrames'] - 1] / blinkVelocity[df['downStroke_del']]
#     df['negAmpVelRatioBase'] = (100 * abs(df['KKK'])) / srate
#
#     ## Blink amplitude-velocity ratio estimated from tent slope
#
#     # TODO TO remove the minus 1 >> df['maxFrames']-1
#     df['pop'] = candidate_signal[df['maxFrames'] - 1] / df['averRightVelocity']
#     df['negAmpVelRatioTent'] = (100 * abs(df['pop'])) / srate
#
#     df['opi'] = candidate_signal[df['maxFrames'] - 1] / df['averLeftVelocity']
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
#         lambda x: np.argmax(candidate_signal[x['leftZero']:x['rightZero'] + 1] >= x['ampThreshhold']), axis=1)
#
#     df['endShut_tzs'] = df.apply(
#         lambda x: np.argmax(candidate_signal[x['leftZero']:x['rightZero'] + 1][x['start_shut_tzs'] + 1:-1] <
#                             shutAmpFraction * x['maxValues']), axis=1)
#
#     ## PLease expect error here, some value maybe zero or lead to empty cell
#     df['endShut_tzs'] = df['endShut_tzs'] + 1  ## temporary  to delete
#     df['timeShutZero'] = df.apply(
#         lambda x: 0 if x['endShut_tzs'] == np.isnan else x['endShut_tzs'] / srate, axis=1)
#
#     ## Time base shut
#
#     df['ampThreshhold_tbs'] = shutAmpFraction * df['maxValues']
#     df['start_shut_tbs'] = df.apply(
#         lambda x: np.argmax(candidate_signal[x['leftBase']:x['rightBase'] + 1] >= x['ampThreshhold_tbs']), axis=1)
#
#     df['endShut_tbs'] = df.apply(
#         lambda x: np.argmax(candidate_signal[x['leftBase']:x['rightBase'] + 1][x['start_shut_tbs']:-1] <
#                             shutAmpFraction * x['maxValues']), axis=1)
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
#
#     df['start_shut_tst'] = df.apply(
#         lambda x: _start_shut(candidate_signal, x['leftXIntercept_int'], x['rightXIntercept_int'], x['ampThreshhold']), axis=1)
#
#     def _end_shut(arr, leftXIntercept, rightXIntercept, start_shut, maxVal, shutAmpFraction):
#         try:
#             return np.argmax(arr[leftXIntercept:rightXIntercept + 1][int(start_shut):-1] <
#                              shutAmpFraction * maxVal)
#         except ValueError:
#             return np.nan
#
#     ###### WIP GOT ISSUE
#
#     df['endShut_tst'] = df.apply(
#         lambda x: _end_shut(candidate_signal, x['leftXIntercept_int'], x['rightXIntercept_int'], x['start_shut_tst'],
#                             x['maxValues'], shutAmpFraction), axis=1)
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
#     filename = 'data_to_selected_blink_pd_viz_MFF.hkl'
#     hkl.dump([candidate_signal, df_res], filename)
#
#
# def viz_fit_blink_pd_approach(candidate_signal=None,df=None):
#     """
#
#     TODO Viz
#
#     https://stackoverflow.com/a/51928241/6446053
#     https://stackoverflow.com/a/38015084/6446053
#     :return:
#     """
#     from eeg_blinks.viz.viz_pd import viz_complete_blink_prop
#     import mne
#
#     if candidate_signal is None:
#         filename = 'data_to_selected_blink_pd_viz_MFF.hkl'
#         candidate_signal, df = hkl.load(filename)
#
#     title = 'sddd'
#     rep = mne.Report(title=title)
#     print(df.dtypes)
#     cols_int = ['rightBase']
#     df[cols_int] = df[cols_int].astype(int)
#     df.to_excel('da.xlsx')
#     fig_good_blink = []
#     fig_bad_blink = []
#     # for index, row in df.iloc[1:].iterrows():
#     for index, row in df.iterrows():
#
#         dfig = viz_complete_blink_prop(candidate_signal, row)
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
#     spath = 'dreport_mff.html'
#     rep.save(spath, overwrite=True, open_browser=False)
#
#
# def open_file():
#     # fname = '/home/cisir4/Documents/rpb/raja_drows_data/S03/P.mff'
#     # fname = "/mnt/d/data_set/drowsy_driving_raja/S3/P.mff"
#     sample_data_folder = mne.datasets.sample.data_path()
#     sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                         'sample_audvis_filt-0-40_raw.fif')
#     raw = mne.io.read_raw_fif(sample_data_raw_file)
#     # raw = mne.io.read_raw_egi(fname, preload=True)
#     # ch_frontal = ('E1', 'E2', 'E3', 'E8', 'E9', 'E10', 'E14', 'E15', 'E16', 'E17', 'E18')
#     # ch_frontal = ('E8',)
#     # raw.pick_channels(ch_frontal)
#     # raw.set_channel_types({i: 'eeg' for i in ch_frontal})
#
#     # raw.pick_types(eeg=True)
#     # raw = raw.load_data()
#     # ch_list = raw.ch_names
#     raw.filter(0.5, 20.5, fir_design='firwin')
#     raw.resample(100)
#     raw.save('raw_audvis_resampled.fif')
#     drange=[f'EEG 00{X}' for X in range (3)]
#     # drange = ['EEG 002']
#     to_drop_ch = list(set(raw.ch_names) - set(drange))
#     raw = raw.drop_channels(to_drop_ch)
#
#     params = default_setting.params
#     # annot_description = kwargs.get('annot_label', 'eye_blink')
#
#     # ch_list = raw.ch_names if include is None else include
#     sfreq = raw.info['sfreq']
#     # raw.plot()
#
#     logging.info('Get the blink position. This may take some time since channel is process at a time ')
#     ch_list = raw.ch_names
#     blinkPositions_list = [get_blink_position(params, sfreq, blink_component=raw.get_data(picks=ch)[0], ch=ch) for ch in
#                            ch_list]
#
#     import hickle as hkl
#     filename = 'data_to_df_compatble_MFF.hkl'
#     hkl.dump([blinkPositions_list, raw, params, sfreq], filename)
#     k = 1
#
#     # return blinkPositions_list, raw, params,sfreq
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
# def _peak_vallay_frame_matlab():
#     g=1
# def prepare_df_compatible(ch_data=None, candidate_signal=None, params=None, srate=None):
#     if ch_data is None:
#         import hickle as hkl
#         filename = 'data_to_df_compatble_MFF.hkl'
#
#         blinkPositions_list, raw, params, srate = hkl.load(filename)
#         ch_data = blinkPositions_list[0]
#         ch = ch_data['ch']
#         print(ch)
#         candidate_signal = raw.get_data(picks=ch)[0]
#
#     startBlinks = ch_data['start_blink']  # ch_data is equivalent to blinkPosition
#     endBlinks = ch_data['end_blink']
#
#     maxValues, maxFrames = zip(*[_get_max_frame(candidate_signal, dstartBlinks, dendBlinks) for
#                                  dstartBlinks, dendBlinks in zip(startBlinks, endBlinks)])
#     ## Calculate the fits
#
#     baseFraction = 0.1  # Fraction from top and bottom
#     maxFrames = np.array(maxFrames)
#     maxValues = np.array(maxValues)
#     outerStarts = np.append(0, maxFrames[0:-1])
#     outerEnds = np.append(maxFrames[1:], candidate_signal.size)
#
#     df = pd.DataFrame(dict(maxFrames=maxFrames, maxValues=maxValues, startBlinks=startBlinks, endBlinks=endBlinks,
#                            outerStarts=outerStarts, outerEnds=outerEnds))
#
#     ## remove blinks that arent separated
#
#     df['blink_duration'] = (df['endBlinks'] - df['startBlinks']) / srate
#
#     df = df[df.blink_duration.ge(0.05)].reset_index(drop=True)
#
#     if ch_data is None:
#         # Data are all store within the same folder of this main file
#         import hickle as hkl
#         filename = 'data_to_get_zero_crossing_pd_MFF.hkl'
#         hkl.dump([candidate_signal, df], filename)
#     else:
#         return df
#
#
# def loop_file():
#     import hickle as hkl
#     filename = 'data_to_df_compatble_MFF.hkl'
#
#     blinkPositions_list, raw, params, srate = hkl.load(filename)
#
#     all_data_info = []
#     all_d = []
#     for bp in blinkPositions_list:
#         ch = bp['ch']
#         print(ch)
#         # signal_eeg = raw.get_data(picks=ch)[0]
#         df = prepare_df_compatible(ch_data=bp, candidate_signal=raw.get_data(picks=ch)[0],
#                                    params=params, srate=srate)
#         df = fit_blink_pd_approach(df=df, candidate_signal=raw.get_data(picks=ch)[0])
#         d = _get_param_for_selection(signal=raw.get_data(picks=ch)[0], df=df, params=params, ch=ch)
#         # df_s=pd.DataFrame(d)
#         all_data_info.append(df)
#         all_d.append(d)
#     j = 1
#     print(df)
#     ch_blink_stat = pd.DataFrame(all_d)
#     ch_selected=get_best_channel(df=ch_blink_stat,params=params)
#     hkl.dump([raw, params, ch_selected, all_data_info], 'data_to_viz_complete.hkl')
#
#     # filename = 'data_to_for_selection_MFF.hkl'
#     # hkl.dump([raw, params, srate, all_data_info], filename)
#
#
# def extracBLinks_reduce_number(df, params):
#     '''
#
#     Reduce the number of candidate signals based on these steps
#     1) Reduce the number of candidate signals based on the blink amp ratios:
#         -params ['params_blinkAmpRange_1'],params ['params_blinkAmpRange_2']
#     2) Find the ones that meet the minimum good blink threshold
#         -params ['params_goodRatioThreshold']
#     3) See if any candidates meet the good blink ratio criteria
#
#     4) Pick the one with the maximum number of good blinks
#
#     '''
#
#     nbest = 5  # NUmber channel to select
#     nbest_force = 2
#     params_blinkAmpRange_1 = params['params_blinkAmpRange_1']
#     params_blinkAmpRange_2 = params['params_blinkAmpRange_2']
#     params_goodRatioThreshold = params['params_goodRatioThreshold']
#     params_minGoodBlinks = params['params_minGoodBlinks']
#
#     # params_blinkAmpRange_1=2.3
#     df['con_blinkAmpRange'] = np.where((df.blinkAmpRatio >= params_blinkAmpRange_1) &
#                                        (df.blinkAmpRatio <= params_blinkAmpRange_2),
#                                        True, False)
#
#     df['con_GoodBlinks'] = np.where((df.numberGoodBlinks > params_minGoodBlinks),
#                                     True, False)
#
#     df['con_GoodRatio'] = np.where((df.goodRatio > params_goodRatioThreshold),
#                                    True, False)
#     df.sort_values(['goodRatio', 'numberGoodBlinks'], ascending=[False, False], inplace=True)
#
#     nblinkAmpRange = df['con_blinkAmpRange'].sum()
#     # ncon_GoodBlinks=df['con_GoodBlinks'].sum()
#     ncon_GoodRatio = df['con_GoodRatio'].sum()
#     x = 1
#
#     # dfS=df[df['con_blinkAmpRange']==True]
#     # dfS=dfS.head(nbest)
#     # step 1
#
#     import warnings
#     #
#     if nblinkAmpRange == 0:
#         warnings.warn(f'Blink amplitude ratio too low than the predeterimined therehold-- may be noise.')
#         logging.INFO('Try to provide channels that being sorted by GoodRatio, GoodBlink, and blinkAmprange')
#
#         if ncon_GoodRatio != 0:
#
#             df = df[df['con_GoodRatio'] == True]
#             df = df.head(nbest_force)
#         else:
#             ## If the goodratio also zero, then give result with the maximum number of blinks, albeit potential noise
#             df = df.head(nbest_force)
#         return df
#
#     # Step 2
#     # Now see if any candidates meet the good blink ratio criteria
#
#     if ncon_GoodRatio == 0:
#         # if goodRatio is zero, then give
#         if nblinkAmpRange != 0:
#             df = df[df['con_blinkAmpRange'] == True]
#             df = df.head(nbest_force)  # and give the maxblink
#             return df
#         else:
#             return df.head(nbest_force)  # give the maxblink
#
#     # If we fulfill the con_GoodRatio,ncon_GoodRatio
#
#     df = df[df['con_GoodRatio'] == True]
#
#     return df.head(nbest)
#
#
# def get_best_channel(df=None, params=None):
#     # WIP
#     if df is None:
#         params = default_setting.params
#         df = hkl.load('all_dict.hkl')
#
#     df = extracBLinks_reduce_number(df, params)
#     df.reset_index(drop=True, inplace=True)
#
#     return df
#
#
# open_file()  # Lets load the file and find the start end of the blink
# # loop_file()
# # #
# # prepare_df_compatible() # Since the objective is to work with pandas, transform all info to pd compatible
#
# # fit_blink_pd_approach() # Working and tally with viz check
#
# # extracBlinkProperties()  # Pending, dont expect any result from this
# # viz_fit_blink_pd_approach()  # This is working
#
#
# ### SHOULD WE WORK WITH MULTIPLE CHANNEL
# # open_file() # Lets load the file and find the start end of the blink
# # loop_file()
#
# # raw, params, ch_selected, all_data_info = hkl.load('data_to_viz_complete.hkl')
#
# # viz_fit_blink_pd_approach(candidate_signal=raw,params=params,ch_selected=ch_selected,ch_info=all_data_info)  # This is working
#

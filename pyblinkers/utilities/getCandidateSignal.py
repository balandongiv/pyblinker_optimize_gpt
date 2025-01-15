import warnings

import numpy as np

# from pyblinkers.utilities.extractBlinkProperties import _goodblink_based_corr_median_std


# class PrepareSelection:
#     def __init__(self, signal=None, df=None, params=None, ch=None):
#         self.signal = signal
#         self.df = df
#         self.params = params
#         self.ch = ch



    # def get_param_for_selection(self):
    #     self.df['rightR2'] = self.df['rightR2'].abs()
    #     blink_mask = np.zeros(self.signal.size, dtype=bool)
    #
    #     for left_zero_y, right_zero_x in zip(self.df['leftZero'] - 1, self.df['rightZero']):
    #         blink_mask[int(left_zero_y):int(right_zero_x)] = True
    #
    #     outside_blink = np.logical_and(self.signal > 0, ~blink_mask)
    #     inside_blink = np.logical_and(self.signal > 0, blink_mask)
    #
    #     blink_amp_ratio = np.mean(self.signal[inside_blink]) / np.mean(self.signal[outside_blink])
    #
    #
    #
    #     R2_top, R3_top, best_median, best_robust_std = _goodblink_based_corr_median_std(self.df, self.params['correlationThresholdTop'])
    #     R2_bot, R3_bot, worst_median, worst_robust_std = _goodblink_based_corr_median_std(self.df,self.params['correlationThresholdBottom'])
    #     true_top = R2_top.values & R3_top.values
    #     true_bot = R2_bot.values & R3_bot.values
    #     good_values = self.df.loc[true_bot, 'maxValue']
    #
    #     all_values = self.df['maxValue']
    #     cutoff = (best_median * worst_robust_std + worst_median * best_robust_std) / (best_robust_std + worst_robust_std)
    #
    #
    #     mask = np.logical_and(all_values <= best_median + 2 * best_robust_std, all_values >= best_median - 2 * best_robust_std)
    #     all_X = np.count_nonzero(mask)
    #
    #     if all_X != 0:
    #         good_ratio = np.sum(np.logical_and(good_values <= best_median + 2 * best_robust_std, good_values >= best_median - 2 * best_robust_std)) / all_X
    #     else:
    #         good_ratio = np.nan
    #
    #     number_good_blinks = true_bot.sum().item()
    #     all_data = [self.ch, blink_amp_ratio, best_median, best_robust_std, cutoff, good_ratio, number_good_blinks, self.df]
    #     header_eb_label = ['ch', 'blinkAmpRatio', 'bestMedian', 'bestRobustStd', 'cutoff', 'goodRatio',
    #                        'numberGoodBlinks']
    #     data_blink = dict(zip(header_eb_label, all_data))
    #     return data_blink



#
# class ChannelSelection:
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
#     def __init__(self, df, params):
#         self.df = df
#         self.params = params
#         self.nbest = 5
#         self.nbest_force = 2
#         self.params_blinkAmpRange_1 = params['blinkAmpRange_1']
#         self.params_blinkAmpRange_2 = params['blinkAmpRange_2']
#         self.params_goodRatioThreshold = params['goodRatioThreshold']
#         self.params_minGoodBlinks = params['minGoodBlinks']
#         self.reduce_candidate_signals()
#         self.pick_best_channel()
#         self.df.reset_index(drop=True, inplace=True)
#
#     def reduce_candidate_signals(self):
#         self.df['con_blinkAmpRange'] = np.where((self.df.blinkAmpRatio >= self.blinkAmpRange_1) &
#                                                 (self.df.blinkAmpRatio <= self.blinkAmpRange_2),
#                                                 True, False)
#
#         self.df['con_GoodBlinks'] = np.where((self.df.numberGoodBlinks > self.minGoodBlinks),
#                                              True, False)
#
#         self.df['con_GoodRatio'] = np.where((self.df.goodRatio > self.goodRatioThreshold),
#                                             True, False)
#         self.df.sort_values(['goodRatio', 'numberGoodBlinks'], ascending=[False, False], inplace=True)
#
#     def pick_best_channel(self):
#
#         # step 1
#         nblinkAmpRange = self.df['con_blinkAmpRange'].sum()
#         ncon_GoodRatio = self.df['con_GoodRatio'].sum()
#
#         if nblinkAmpRange == 0:
#             warnings.warn(f'Blink amplitude ratio too low than the predeterimined therehold-- may be noise.')
#             if ncon_GoodRatio != 0:
#                 self.df = self.df[self.df['con_GoodRatio'] == True]
#                 self.df = self.df.head(self.nbest_force)
#             else:
#                 self.df = self.df.head(self.nbest_force)
#             return self.df
#
#         # Step 2
#         # Now see if any candidates meet the good blink ratio criteria
#
#         if ncon_GoodRatio == 0:
#             if nblinkAmpRange != 0:
#                 self.df = self.df[self.df['con_blinkAmpRange'] == True]
#                 self.df = self.df.head(self.nbest_force)
#                 return self.df
#             else:
#                 return self.df.head(self.nbest_force)
#
#         # If we fulfill the con_GoodRatio,ncon_GoodRatio
#         df1 = self.df[(self.df['con_GoodRatio'] == True) & (self.df['con_blinkAmpRange'] == True)]
#         if df1.empty:
#             """
#             Most probably we have
#             con_blinkAmpRange   con_goodBLinks  con_GoodRatio
#             FALSE               TRUE            TRUE
#             TRUE                TRUE            FALSE
#             """
#             self.df = self.df[(self.df['con_GoodRatio'] == True) & (self.df['con_GoodBlinks'] == True)]
#             return self.df
#         else:
#             """
#             Ideal case
#             """
#             self.df = self.df[(self.df['con_GoodRatio'] == True) & (self.df['con_blinkAmpRange'] == True) & (self.df['con_GoodBlinks'] == True)]
#
#             return self.df.head(self.nbest)
#

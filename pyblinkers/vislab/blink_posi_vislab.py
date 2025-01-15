import logging
import numpy as np
import pandas as pd
from pyblinkers.vislab.getBlinkPositions_vislab import getBlinkPosition,getBlinkPositionPython
from pyblinkers.utilities.zero_crossing import left_right_zero_crossing

logging.basicConfig(level=logging.INFO)

# def _get_max_frame(data, startBlinks, endBlinks):
#     blinkRange = np.arange(startBlinks, endBlinks + 1)
#     blink_frame = data[startBlinks:endBlinks + 1]
#     maxValues = np.amax(blink_frame)
#     maxFrames = blinkRange[np.argmax(blink_frame)]
#     return maxValues, maxFrames


# class extractBlinks:
#     def __init__(self, data, sfreq, params, ch):
#         self.data = data
#         self.sfreq = sfreq
#         self.params = params
#         self.ch = ch
#         self.srate = sfreq
#
#     def getBlinksCoordinate(self):
#
#         # getBlinkPositions (STEP 1bi)
#         self.ch_data = getBlinkPosition(self.params, blinkComp=self.data, ch=self.ch)
#
#         # While I am interested to migrate to python, I will keep the below line commented as the error is not resolved
#         # self.ch_data = getBlinkPositionPython(self.params, blinkComp=self.data, ch=self.ch)
#
#         self.startBlinks = self.ch_data['startBlinks'].to_numpy()
#         self.endBlinks = self.ch_data['endBlinks'].to_numpy()
#
#         # self._get_blink_position()
#         self._get_max_frame_values()
#         self.blink_frame = pd.DataFrame(dict(maxFrame=self.maxFrames, maxValue=self.maxValues,
#                                              startBlinks=self.startBlinks, endBlinks=self.endBlinks,
#                                              outerStarts=self.outerStarts, outerEnds=self.outerEnds))
#
#         self._get_zero_crossing()
#
#         return self.blink_frame



    # def _get_max_frame(self, startBlinks, endBlinks):
    #     blinkRange = np.arange(startBlinks, endBlinks + 1)
    #     blink_frame = self.data[startBlinks:endBlinks + 1]
    #     maxValues = np.amax(blink_frame)
    #     maxFrames = blinkRange[np.argmax(blink_frame)]
    #     return maxValues, maxFrames

    # def _get_max_frame_values(self):
    #     maxValues, maxFrames = zip(*[_get_max_frame(self.data,dstartBlinks, dendBlinks) for
    #                                  dstartBlinks, dendBlinks in zip(self.startBlinks, self.endBlinks)])
    #     self.maxFrames = np.array(maxFrames)
    #     self.maxValues = np.array(maxValues)
    #     self.outerStarts = np.append(0, self.maxFrames[0:-1])
    #     self.outerEnds = np.append(self.maxFrames[1:], self.data.size)
    #     g=1

    # def _compute_duration(self):
    #     pass
    #     # self.blink_frame['blink_duration'] = (self.blink_frame['endBlinks'] - self.blink_frame['startBlinks']) / self.srate

    # def _filter_duration(self):
    #     self.blink_frame = self.blink_frame[self.blink_frame.blink_duration.ge(0.05)].reset_index(drop=True)
    #


    # def _get_zero_crossing(self):
    #     self.blink_frame[['leftZero', 'rightZero']] = self.blink_frame.apply(lambda x: left_right_zero_crossing(self.data,x['maxFrame'], x['outerStarts'],
    #                                                                                     x['outerEnds']), axis=1,
    #                                                  result_type="expand")
    #
    #



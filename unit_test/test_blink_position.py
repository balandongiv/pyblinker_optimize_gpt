import pickle
import unittest

import numpy as np
import pandas as pd

# from unit_test.develop_blink_position import get_blink_position
from pyblinkers.getBlinkPositions import get_blink_position

class TestGetBlinkPosition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load debug data once for all tests
        with open('file_test_blink_position.pkl', 'rb') as f:
            cls.debug_data = pickle.load(f)

    def test_blink_detection(self):
        params = self.debug_data['params']
        blink_component = self.debug_data['blink_component']
        ch = self.debug_data['ch']
        threshold = self.debug_data['threshold']
        min_blink_frames = self.debug_data['min_blink_frames']
        expected_output = self.debug_data['output']

        # Run the function
        result = get_blink_position(
            params=params,
            blink_component=blink_component,
            ch=ch,
            threshold=threshold,
            min_blink_frames=min_blink_frames
        )

        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that the result has the same columns
        self.assertListEqual(list(result.columns), ['startBlinks', 'endBlinks'])

        # Check that the values are the same (both start and end)
        np.testing.assert_array_equal(result['startBlinks'].values, expected_output['startBlinks'].values)
        np.testing.assert_array_equal(result['endBlinks'].values, expected_output['endBlinks'].values)

if __name__ == '__main__':
    unittest.main()
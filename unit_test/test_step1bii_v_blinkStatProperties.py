import unittest
import numpy as np
import pandas as pd
from pyblinkers.extractBlinkProperties import get_blink_statistic
from unit_test.debugging_tools import load_matlab_data


class TestBlinkStatistic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by loading input and ground truth data.
        """
        cls.mat_file_path_input = r'..\Devel\step1bii_v_input_blinkStatProperties.mat'
        cls.mat_file_path_output = r'..\Devel\step1bii_v_output_blinkStatProperties.mat'

        # Load data
        input_data, output_datax = load_matlab_data(cls.mat_file_path_input, cls.mat_file_path_output)
        cls.input_data = input_data
        cls.output_data = output_datax

        # Candidate signals
        cls.signal = cls.input_data['candidateSignals']

        # Blink fits as DataFrame
        cls.df = pd.DataFrame.from_records(cls.input_data['blinkFits'])

        # Ground truth signal data
        cls.signalData_gt = cls.output_data['blinks']['signalData']

        # Remove unwanted keys from the ground truth for comparison
        for key in ["signal", "blinkPositions", "signalType", "signalNumber", "signalLabel"]:
            cls.signalData_gt.pop(key, None)

        # Use fixed zThresholds
        cls.zThresholds = np.array([[0.9, 0.98], [2.0, 5.0]])

    def test_blink_statistic(self):
        """
        Test the get_blink_statistic function output against the MATLAB ground truth.
        """
        # Compute blink statistics
        signalData = get_blink_statistic(self.df, self.zThresholds, signal=self.signal)

        # Check for differences between the dictionaries
        differences = {}
        for key in self.signalData_gt.keys():
            if key not in signalData:
                differences[key] = f"Key '{key}' is missing in the computed signalData."
            elif not np.allclose(self.signalData_gt[key], signalData[key], atol=1e-6, equal_nan=True):
                differences[key] = {
                    'ground_truth': self.signalData_gt[key],
                    'computed': signalData[key]
                }

        for key in signalData.keys():
            if key not in self.signalData_gt:
                differences[key] = f"Key '{key}' is missing in the ground truth signalData."

        # Log differences if any
        if differences:
            print("\nDifferences found in signalData:")
            for key, diff in differences.items():
                print(f"Key: {key}, Difference: {diff}")

        # Assert no differences
        self.assertFalse(
            differences,
            f"The computed signalData does not match the ground truth. Differences: {differences}"
        )


if __name__ == '__main__':
    unittest.main()

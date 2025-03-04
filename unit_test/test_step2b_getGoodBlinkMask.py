import unittest
import numpy as np
import pandas as pd
from unit_test.debugging_tools import load_matlab_data
from pyblinkers.extractBlinkProperties import get_good_blink_mask


class TestGetGoodBlinkMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by loading input and ground truth candidate_signal.
        """
        cls.mat_file_path_input = r'..\migration_files\step2b_data_input_getGoodBlinkMask.mat'
        cls.mat_file_path_output = r'..\migration_files\step2b_data_output_getGoodBlinkMask.mat'

        # Load candidate_signal
        input_data, output_datax = load_matlab_data(cls.mat_file_path_input, cls.mat_file_path_output)
        cls.input_data = input_data
        cls.goodblinkmask_output = output_datax['goodBlinkMask'].astype(bool)

        # Blink fits as DataFrame
        cls.blinkFits = pd.DataFrame.from_records(cls.input_data['blinkFits'])

        # Use fixed z_thresholds instead of MATLAB values
        cls.zThresholds = np.array([[0.9, 0.98], [2.0, 5.0]])

    def test_good_blink_mask(self):
        """
        Test the get_good_blink_mask function output against the MATLAB ground truth.
        """
        # Compute good blink mask and selected DataFrame
        goodBlinkMask, selected_df = get_good_blink_mask(
            self.blinkFits,
            self.input_data['specifiedMedian'],
            self.input_data['specifiedStd'],
            self.zThresholds
        )

        # Convert results to DataFrame for comparison
        comparison_df = pd.DataFrame({
            'goodBlinkMask': goodBlinkMask,
            'goodblinkmask_output': self.goodblinkmask_output
        })

        # Check for inconsistencies
        inconsistent = comparison_df.apply(
            lambda row: row['goodBlinkMask'] != row['goodblinkmask_output'], axis=1
        ).any()

        # Log comparison details if inconsistencies exist
        if inconsistent:
            print("\nInconsistent Rows in Good Blink Mask:")
            print(comparison_df[comparison_df['goodBlinkMask'] != comparison_df['goodblinkmask_output']])

        # Assert arrays are the same
        self.assertTrue(
            np.array_equal(goodBlinkMask, self.goodblinkmask_output),
            "The calculated goodBlinkMask does not match the MATLAB output."
        )


if __name__ == '__main__':
    unittest.main()

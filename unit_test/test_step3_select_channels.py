import unittest
import logging
import pandas as pd
from pyblinkers.getRepresentativeChannel import (
    filter_blink_amplitude_ratios,
    filter_good_blinks,
    filter_good_ratio,
    select_max_good_blinks
)
from unit_test.debugging_tools import load_matlab_data
from pyblinkers import default_setting

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSelectChannelCompact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by loading input and ground truth data.
        """
        cls.mat_file_path_input = r'..\Devel\step3a_input_selectChannel_compact.mat'
        cls.mat_file_path_output = r'..\Devel\step3a_output_selectChannel_compact.mat'

        # Load data
        input_data, output_data = load_matlab_data(
            input_path=cls.mat_file_path_input,
            output_path=cls.mat_file_path_output
        )
        cls.input_data = input_data
        cls.output_data = output_data

        # Ground truth signal data
        cls.signal_data_gt = pd.DataFrame.from_records(cls.output_data['blinks']['signalData'])
        cls.signal_data_gt = cls.signal_data_gt.drop(columns=['signal', 'blinkPositions', 'signalType', 'signalNumber'])
        cls.signal_data_gt = cls.signal_data_gt.rename(columns={'signalLabel': 'ch'})

        # Signal data for processing
        cls.signal_data = pd.DataFrame.from_records(cls.input_data['signalData'])
        cls.signal_data = cls.signal_data.drop(columns=['signal', 'blinkPositions', 'signalType', 'signalNumber'])
        cls.signal_data = cls.signal_data.rename(columns={'signalLabel': 'ch'})

        # Parameters
        cls.params = default_setting.params

    def test_select_channel_compact(self):
        """
        Test the blink signal selection process against MATLAB ground truth.
        """
        # Apply the blink signal selection process
        df = self.signal_data.copy()
        df = filter_blink_amplitude_ratios(df, self.params)
        df = filter_good_blinks(df, self.params)
        df = filter_good_ratio(df, self.params)
        signal_data_output = select_max_good_blinks(df)

        # Columns to ignore
        columns_to_ignore = ['status', 'select']

        # Log the removal of columns
        logger.info(
            "Removing the following columns from comparison: %s", columns_to_ignore
        )

        # Remove `status` and `select` columns from the comparison
        signal_data_output = signal_data_output.drop(columns=columns_to_ignore, errors='ignore')
        self.signal_data_gt = self.signal_data_gt.drop(columns=columns_to_ignore, errors='ignore')

        # Sort both DataFrames by 'ch' column
        signal_data_output = signal_data_output.sort_values(by='ch').reset_index(drop=True)
        self.signal_data_gt = self.signal_data_gt.sort_values(by='ch').reset_index(drop=True)

        # Check for differences between the DataFrames
        comparison_report = self.signal_data_gt.compare(signal_data_output, align_axis=1)

        # Log differences if any
        if not comparison_report.empty:
            logger.info("\nDifferences found in signal data output:")
            logger.info(comparison_report)

        # Assert no differences
        self.assertTrue(
            comparison_report.empty,
            f"The processed signal data does not match the ground truth. Differences:\n{comparison_report}"
        )


if __name__ == '__main__':
    unittest.main()

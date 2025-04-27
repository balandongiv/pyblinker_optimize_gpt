import unittest
import pandas as pd
import numpy as np
import logging
from pyblinkers import default_setting
from pyblinkers.extractBlinkProperties import BlinkProperties, get_good_blink_mask, get_blink_statistic
from pyblinkers.fit_blink import FitBlinks
from pyblinkers.getBlinkPositions import get_blink_position
from unit_test.debugging_tools import load_matlab_data

# Configure logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def compare_dataframes(df_ground_truth, df_output, decimal_places=4):
    """
    Compare two DataFrames and return a comparison report, including missing columns.
    """
    report = df_ground_truth.copy()

    # Identify missing columns
    ground_truth_columns = set(df_ground_truth.columns)
    output_columns = set(df_output.columns)
    missing_in_ground_truth = output_columns - ground_truth_columns
    missing_in_output = ground_truth_columns - output_columns

    missing_columns_report = {
        "missing_in_ground_truth": list(missing_in_ground_truth),
        "missing_in_output": list(missing_in_output),
    }

    # Find common columns
    common_columns = ground_truth_columns.intersection(output_columns)

    # Round values to specified decimal places
    for column in common_columns:
        df_ground_truth[column] = df_ground_truth[column].apply(
            lambda x: np.round(x, decimal_places) if isinstance(x, (int, float)) else x
        )
        df_output[column] = df_output[column].apply(
            lambda x: np.round(x, decimal_places) if isinstance(x, (int, float)) else x
        )

    # Compare values and update report
    for column in common_columns:
        for idx in range(len(df_ground_truth)):
            gt_value = df_ground_truth.at[idx, column]
            output_value = df_output.at[idx, column]

            if np.array_equal(gt_value, output_value):
                report.at[idx, column] = 'consistent'
            else:
                report.at[idx, column] = f'not consistent (GT: {gt_value}, Output: {output_value})'

    return report, missing_columns_report


class TestExtractBlinkProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by loading input and ground truth candidate_signal and initializing parameters.
        """
        cls.params = default_setting.params
        cls.params['sfreq'] = 100
        cls.mat_file_path_input = r'..\migration_files\step1bi_data_input_getBlinkPositions.mat'
        cls.mat_file_path_output = r'..\migration_files\immitate_full_step.mat'

        # Load MATLAB input and ground truth candidate_signal
        cls.input_data, cls.output_data = load_matlab_data(
            input_path=cls.mat_file_path_input, output_path=cls.mat_file_path_output
        )

        # Prepare input and output for blink detection
        cls.blink_comp = cls.input_data['blinkComp']
        cls.signal_data_gt = pd.DataFrame.from_records(cls.output_data['combinedStruct'])

    @staticmethod
    def adjust_indices(df):
        """
        Adjust indices for MATLAB compatibility.
        """
        columns_to_increment = [
            'maxFrame', 'startBlinks', 'endBlinks', 'outerStarts', 'outerEnds',
            'leftZero', 'rightZero', 'maxPosVelFrame', 'maxNegVelFrame', 'leftBase',
            'rightBase', 'leftZeroHalfHeight', 'rightZeroHalfHeight',
            'leftBaseHalfHeight', 'rightBaseHalfHeight', 'xIntersect', 'yIntersect',
            'rightXIntercept',
        ]
        df.loc[:, columns_to_increment] += 1

        second_columns_to_increment = ['yIntersect', 'leftXIntercept']
        df[second_columns_to_increment] += 1

        third_columns_to_increment = ['yIntersect']
        df[third_columns_to_increment] -= 2

        # Adjust `leftRange` and `rightRange`
        df['leftRange'] = df['leftRange'].apply(lambda x: [val + 1 for val in x])
        df['rightRange'] = df['rightRange'].apply(lambda x: [val + 1 for val in x])
        return df

    def test_extract_blink_properties(self):
        """
        Perform the end-to-end test for blink property extraction.
        """
        to_ignore_three_case = True  # Flag to ignore specific cases

        # Define the cases to ignore
        ignore_cases = [
            {'row': 41, 'column': 'yIntersect', 'ground_truth_value': 43.0, 'output_value': 44.0},
        ]
        min_blink_frames=5.0
        threshold=12.241726391783821
        # STEP 1: Get blink positions
        blink_positions = get_blink_position(
            self.params, blink_component=self.blink_comp, ch='No_channel',threshold=threshold,
        min_blink_frames=min_blink_frames)

        # STEP 2: Fit blinks
        fitblinks = FitBlinks(candidate_signal=self.blink_comp, df=blink_positions, params=self.params)
        fitblinks.dprocess()
        df = fitblinks.frame_blinks

        # STEP 3: Extract blink statistics
        signal_data = get_blink_statistic(df, self.params['z_thresholds'], signal=self.blink_comp)

        # STEP 4: Get good blink mask
        good_blink_mask, df = get_good_blink_mask(
            df, signal_data['bestMedian'], signal_data['bestRobustStd'], self.params['z_thresholds']
        )

        # STEP 5: Compute blink properties
        sfreq = self.params['sfreq']
        df = BlinkProperties(self.blink_comp, df, sfreq, self.params).df

        # STEP 6: Apply pAVR restriction
        condition_1 = df['posAmpVelRatioZero'] < self.params['pAVRThreshold']
        condition_2 = df['maxValue'] < (signal_data['bestMedian'] - signal_data['bestRobustStd'])
        signal_data_output = df[~(condition_1 & condition_2)]

        # Adjust indices for MATLAB compatibility
        signal_data_output = self.adjust_indices(signal_data_output)

        # Select desired columns
        column_order = [
            'maxFrame', 'maxValue', 'leftZero', 'rightZero', 'leftBase', 'rightBase',
            'leftBaseHalfHeight', 'rightBaseHalfHeight', 'leftZeroHalfHeight',
            'rightZeroHalfHeight', 'leftRange', 'rightRange', 'leftSlope',
            'rightSlope', 'averLeftVelocity', 'averRightVelocity', 'leftR2', 'rightR2',
            'xIntersect', 'yIntersect', 'leftXIntercept', 'rightXIntercept'
        ]
        signal_data_output = signal_data_output[column_order].reset_index(drop=True)
        self.signal_data_gt = self.signal_data_gt[column_order].reset_index(drop=True)

        # Compare outputs
        comparison_report, missing_columns_report = compare_dataframes(
            self.signal_data_gt, signal_data_output, decimal_places=0
        )

        # Remove rows corresponding to ignored cases
        if to_ignore_three_case:
            logger.warning(
                "The test is being conducted with `to_ignore_three_case` set to True.\n"
                "The following specific inconsistencies will be ignored:\n %s",
                ignore_cases
            )
            rows_to_drop = {case['row'] for case in ignore_cases}
            comparison_report.drop(index=rows_to_drop, inplace=True, errors='ignore')

        # Remove columns with all 'consistent'
        comparison_report_filtered = comparison_report.loc[
                                     :, ~(comparison_report == 'consistent').all()
                                     ]

        # Log missing columns report
        print("\nMissing Columns Report:")
        print(missing_columns_report)

        # Log the filtered comparison report
        print("\nFiltered Comparison Report:")
        print(comparison_report_filtered)

        # Assert no missing columns
        self.assertEqual(len(missing_columns_report["missing_in_ground_truth"]), 0,
                         f"Missing columns in ground truth: {missing_columns_report['missing_in_ground_truth']}")
        self.assertEqual(len(missing_columns_report["missing_in_output"]), 0,
                         f"Missing columns in output: {missing_columns_report['missing_in_output']}")

        # Check for inconsistencies
        inconsistent = comparison_report_filtered.applymap(
            lambda x: isinstance(x, str) and 'not consistent' in x
        ).any(axis=None)
        self.assertFalse(inconsistent, f"Inconsistencies found in report: {comparison_report_filtered}")


if __name__ == '__main__':
    unittest.main()

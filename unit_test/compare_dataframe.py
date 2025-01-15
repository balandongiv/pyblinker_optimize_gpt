# import numpy as np
#
#
# # Function to compare dataframes
# def compare_dataframes(df_ground_truth, df_output, decimal_places=3):
#     """
#     Compares columns of the same name between df_ground_truth and df_output
#     and generates a report indicating whether the values are consistent or not.
#
#     Args:
#     - df_ground_truth (pd.DataFrame): Ground truth dataframe
#     - df_output (pd.DataFrame): Output dataframe to be compared with ground truth
#     - decimal_places (int): Number of decimal places for comparison (default is 3)
#
#     Returns:
#     - pd.DataFrame: A report dataframe with 'consistent' or 'not consistent' for each cell comparison.
#     - dict: A report on missing columns in either dataframe.
#     """
#     # Ensure both dataframes have the 'maxFrame' column
#     if 'maxFrame' not in df_ground_truth or 'maxFrame' not in df_output:
#         raise ValueError("Both dataframes must have the 'maxFrame' column.")
#
#     # Create an empty report dataframe that has the same structure as df_ground_truth
#     report = df_ground_truth.copy()
#
#     # Find common columns between both dataframes (excluding 'maxFrame')
#     common_columns = get_common_columns(df_ground_truth, df_output)
#
#     # Identify and report missing columns
#     missing_columns_report = get_missing_columns_report(df_ground_truth, df_output)
#
#     # Round the relevant columns
#     round_columns(df_ground_truth, df_output, common_columns, decimal_places)
#
#     # Compare values and update report
#     compare_column_values(df_ground_truth, df_output, report, common_columns)
#
#     # Retain the 'maxFrame' column as is in the report
#     report['maxFrame'] = df_ground_truth['maxFrame']
#
#     return report, missing_columns_report
#
# # Helper function to get common columns excluding 'maxFrame'
# def get_common_columns(df_ground_truth, df_output):
#     return set(df_ground_truth.columns).intersection(set(df_output.columns)) - {'maxFrame'}
#
# # Helper function to identify and report missing columns
# def get_missing_columns_report(df_ground_truth, df_output):
#     gt_columns = set(df_ground_truth.columns)
#     output_columns = set(df_output.columns)
#
#     missing_in_gt = output_columns - gt_columns
#     missing_in_output = gt_columns - output_columns
#
#     return {
#         'missing_in_ground_truth': list(missing_in_gt),
#         'missing_in_output': list(missing_in_output)
#     }
#
# # Helper function to round the values in common columns
# def round_columns(df_ground_truth, df_output, common_columns, decimal_places):
#     for column in common_columns:
#         # Round the ground truth dataframe values
#         df_ground_truth[column] = df_ground_truth[column].apply(
#             lambda x: np.round(x, decimal_places) if isinstance(x, (int, float)) else np.round(np.array(x), decimal_places)
#         )
#
#         # Round the output dataframe values
#         df_output[column] = df_output[column].apply(
#             lambda x: np.round(x, decimal_places) if isinstance(x, (int, float)) else np.round(np.array(x), decimal_places)
#         )
#
# # Helper function to compare the values of common columns
# def compare_column_values(df_ground_truth, df_output, report, common_columns):
#     for column in common_columns:
#         for idx in range(len(df_ground_truth)):
#             gt_value = df_ground_truth.at[idx, column]
#             output_value = df_output.at[idx, column]
#
#             # Compare the values after rounding
#             if np.array_equal(gt_value, output_value):
#                 report.at[idx, column] = 'consistent'
#             else:
#                 report.at[idx, column] = f'not consistent (GT: {gt_value}, Output: {output_value})'

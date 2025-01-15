from pyblinkers.getRepresentativeChannel import filter_blink_amplitude, filter_good_blinks, filter_good_ratio, select_max_good_blinks


import unittest

class TestBlinkProcessing(unittest.TestCase):

    def setUp(self):
        """Setup sample data and parameters for tests."""
        self.data = {
            'blinkAmpRatios': [0.1, 0.5, 0.9, 1.5, 0.2],
            'numberGoodBlinks': [5, 15, 2, 1, 10],
            'goodRatio': [0.8, 0.9, 0.6, 0.3, 0.7]
        }
        self.df = pd.DataFrame(self.data)
        self.params = {
            'params_blinkAmpRange_1': 0.4,
            'params_blinkAmpRange_2': 1.0,
            'minGoodBlinks': 5,
            'goodRatioThreshold': 0.7
        }

    def test_blink_amplitude_filter(self):
        """Test filtering by blink amplitude range."""
        result_df = filter_blink_amplitude(self.df.copy(), self.params)
        filtered_rows = result_df[(result_df['blinkAmpRatios'] >= self.params['params_blinkAmpRange_1']) &
                                  (result_df['blinkAmpRatios'] <= self.params['params_blinkAmpRange_2'])]
        if filtered_rows.empty:
            self.assertEqual(result_df['status'].iloc[0], "Blink amplitude too low -- may be noise")
        else:
            self.assertTrue((filtered_rows['blinkAmpRatios'] >= self.params['params_blinkAmpRange_1']).all())
            self.assertTrue((filtered_rows['blinkAmpRatios'] <= self.params['params_blinkAmpRange_2']).all())

    def test_good_blinks_filter(self):
        """Test filtering by number of good blinks."""
        result_df = filter_good_blinks(self.df.copy(), self.params)
        filtered_rows = result_df[result_df['numberGoodBlinks'] > self.params['minGoodBlinks']]
        if filtered_rows.empty:
            self.assertEqual(result_df['status'].iloc[0],
                             f"Fewer than {self.params['minGoodBlinks']} minGoodBlinks were found")
        else:
            self.assertTrue((filtered_rows['numberGoodBlinks'] > self.params['minGoodBlinks']).all())

    def test_good_ratio_filter(self):
        """Test filtering by good ratio threshold."""
        result_df = filter_good_ratio(self.df.copy(), self.params)
        filtered_rows = result_df[result_df['goodRatio'] >= self.params['goodRatioThreshold']]
        if filtered_rows.empty:
            self.assertEqual(result_df['status'].iloc[0], "Good ratio too low")
        else:
            self.assertTrue((filtered_rows['goodRatio'] >= self.params['goodRatioThreshold']).all())

    def test_select_max_good_blinks(self):
        """Test selecting the row with maximum numberGoodBlinks."""
        result_df = select_max_good_blinks(self.df.copy())
        max_idx = self.df['numberGoodBlinks'].idxmax()
        self.assertTrue(result_df.loc[max_idx, 'select'])
        self.assertFalse(result_df.loc[result_df.index != max_idx, 'select'].any())

    def test_full_processing(self):
        """Test the full processing pipeline."""
        result_df = filter_blink_amplitude(self.df.copy(), self.params)
        result_df = filter_good_blinks(result_df, self.params)
        result_df = filter_good_ratio(result_df, self.params)
        result_df = select_max_good_blinks(result_df)
        max_idx = self.df['numberGoodBlinks'].idxmax()
        self.assertTrue(result_df.loc[max_idx, 'select'])
        self.assertFalse(result_df.loc[result_df.index != max_idx, 'select'].any())


# Run the tests
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestBlinkProcessing))



# df = filter_blink_amplitude(df, params)
# df = filter_good_blinks(df, params)
# df = filter_good_ratio(df, params)
# df = select_max_good_blinks(df)
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

# Assuming the function get_blink_position is in a file named blink_detector.py
# from blink_detector import get_blink_position
# For self-contained testing, let's paste the function here:
def get_blink_position(
        params: dict,
        blink_component: np.ndarray | None = None,
        ch: str | None = None, # Not used, for compatibility
        threshold: float | None = None,
        min_blink_frames: int | None = None,
) -> pd.DataFrame:
    """
    Detect blink start- and end-frame indices in a 1-D blink component.

    Parameters
    ----------
    params : dict
        Must contain:
        • 'sfreq' (float) – sampling rate in Hz
        • 'minEventLen' (float) – minimum gap allowed between two blinks (s)
    blink_component : 1-D ndarray
        Independent component or channel dominated by eye blinks.
    ch : str, optional
        Channel name (kept for compatibility; no longer used internally).
    threshold : float
        Amplitude threshold that defines “inside a blink”.
    min_blink_frames : int
        Minimum number of consecutive frames over threshold to count as a blink.
        Specifically, the number of frames (duration) must be > min_blink_frames.

    Returns
    -------
    pd.DataFrame with columns ['startBlinks', 'endBlinks']
    """
    if blink_component is None:
        raise ValueError("blink_component must be provided")
    if threshold is None or min_blink_frames is None:
        raise ValueError("threshold and min_blink_frames must be provided")
    if "sfreq" not in params or "minEventLen" not in params:
        raise KeyError("params must contain 'sfreq' and 'minEventLen'")


    # --- 1. binary mask of “blink” frames -----------------------------------
    mask = blink_component > threshold

    # --- 2. locate rising & falling edges -----------------------------------
    diff = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(diff == 1) + 1          # +1 because diff is shifted
    ends   = np.flatnonzero(diff == -1) + 1          # +1 because diff is shifted, end is exclusive

    # handle blink starting at index 0 or ending at last sample
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1] and ends.size > 0 and ends[-1] < mask.size : # only append if last segment is a blink
        ends = np.append(ends, mask.size)
    elif mask[-1] and ends.size == 0 and starts.size > 0: # Special case: signal is all blink
        ends = np.append(ends, mask.size)
    elif mask[-1] and starts.size > 0 and starts[-1] > (ends[-1] if ends.size >0 else -1): # blink ends at last sample
        ends = np.append(ends, mask.size)


    if starts.size == 0 or ends.size == 0:
        # No complete blink found
        return pd.DataFrame({'startBlinks': [], 'endBlinks': []}, dtype=int)

    # Ensure starts and ends are paired correctly after edge handling
    # A start must be followed by an end
    if ends[0] < starts[0]: # First end is before first start (e.g. signal starts with end of a blink)
        ends = ends[1:]
    if starts.size > ends.size: # More starts than ends (e.g. signal ends mid-blink)
        starts = starts[:ends.size] # Truncate starts to match ends
    elif ends.size > starts.size: # More ends than starts
        ends = ends[:starts.size] # Truncate ends to match starts

    if starts.size == 0:
        return pd.DataFrame({'startBlinks': [], 'endBlinks': []}, dtype=int)


    # --- 3. enforce minimum blink duration ----------------------------------
    # duration is end (exclusive) - start (inclusive)
    durations = ends - starts
    valid = durations > min_blink_frames # Strictly greater
    starts, ends = starts[valid], ends[valid]

    if starts.size == 0:
        return pd.DataFrame({'startBlinks': [], 'endBlinks': []}, dtype=int)

    # --- 4. remove blinks that are too close together -----------------------
    sfreq = params["sfreq"]
    min_gap_sec = params["minEventLen"]
    min_gap_frames = min_gap_sec * sfreq

    if starts.size <= 1: # Not enough blinks to check for gaps
        return pd.DataFrame({'startBlinks': starts, 'endBlinks': ends})

    # gap between current start and previous end (in frames)
    # `ends[:-1]` is the end of the previous blink (exclusive)
    # `starts[1:]` is the start of the current blink (inclusive)
    # So, `starts[1:] - ends[:-1]` is the number of frames between blinks
    gaps = starts[1:] - ends[:-1]
    too_close = gaps <= min_gap_frames # If gap is less than or equal to min_gap, it's too close

    # mark both members of each too-close pair for deletion
    keep = np.ones_like(starts, dtype=bool)
    idx = np.where(too_close)[0] # These are indices into the `gaps` array
    # which correspond to the *second* blink in the pair
    keep[idx] = False          # Mark previous blink (indexed by `idx` in `starts`/`ends`)
    keep[idx + 1] = False      # Mark next blink (indexed by `idx + 1` in `starts`/`ends`)

    starts, ends = starts[keep], ends[keep]

    return pd.DataFrame({'startBlinks': starts, 'endBlinks': ends})


class TestGetBlinkPosition(unittest.TestCase):

    def setUp(self):
        self.sfreq = 100  # Hz
        self.min_event_len_sec = 0.5  # seconds, so 50 frames gap
        self.params = {"sfreq": self.sfreq, "minEventLen": self.min_event_len_sec}
        self.threshold = 5.0
        self.min_blink_frames = 3 # A blink must have > 3 frames (i.e., at least 4 frames)

    def assert_dataframes_equal(self, df1, df2, msg=None):
        """Helper to assert two DataFrames are equal, handling empty cases and dtypes."""
        if df1.empty and df2.empty:
            # Check columns if both are empty
            self.assertListEqual(list(df1.columns), list(df2.columns), msg)
            # Check dtypes for empty dataframes
            if not df1.empty: # Should not happen if both empty
                for col in df1.columns:
                    self.assertEqual(df1[col].dtype, df2[col].dtype, f"Dtype mismatch for col {col}: {msg}")
            return

        # For non-empty, assert_frame_equal does a good job
        # Ensure integer dtypes for comparison if not empty, as output should be int
        if not df1.empty:
            df1 = df1.astype(int)
        if not df2.empty:
            df2 = df2.astype(int)
        assert_frame_equal(df1, df2, check_dtype=False, rtol=0, atol=0, obj=msg)


    def test_no_blinks_signal_always_below_threshold(self):
        blink_component = np.array([1.0, 2.0, 1.0, 3.0, 0.0])
        expected_df = pd.DataFrame({'startBlinks': [], 'endBlinks': []}, dtype=int)
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: No blinks, signal always below threshold")

    def test_single_valid_blink(self):
        # Blink duration: 10-3 = 7 frames. 7 > min_blink_frames (3) -> True
        blink_component = np.array([0,0,0, 6,6,6,6,6,6,6, 0,0,0])
        #                            0 1 2  3 4 5 6 7 8 9 10 11 12
        # Start at 3, end at 10 (exclusive)
        expected_df = pd.DataFrame({'startBlinks': [3], 'endBlinks': [10]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Single valid blink")

    def test_blink_too_short(self):
        # Blink duration: 5-3 = 2 frames. 2 > min_blink_frames (3) -> False
        blink_component = np.array([0,0,0, 6,6, 0,0,0])
        #                            0 1 2  3 4  5 6 7
        # Potential start at 3, end at 5 (exclusive)
        expected_df = pd.DataFrame({'startBlinks': [], 'endBlinks': []}, dtype=int)
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames) # Needs > 3 frames
        self.assert_dataframes_equal(result_df, expected_df, "Test: Blink too short")

    def test_blink_just_long_enough(self):
        # Blink duration: 7-3 = 4 frames. 4 > min_blink_frames (3) -> True
        blink_component = np.array([0,0,0, 6,6,6,6, 0,0,0])
        #                            0 1 2  3 4 5 6  7 8 9
        # Start at 3, end at 7 (exclusive)
        expected_df = pd.DataFrame({'startBlinks': [3], 'endBlinks': [7]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames) # Needs > 3 frames
        self.assert_dataframes_equal(result_df, expected_df, "Test: Blink just long enough")

    def test_blink_at_start_of_signal(self):
        # Blink duration: 0 to 5 (exclusive) = 5 frames. 5 > 3 -> True
        blink_component = np.array([6,6,6,6,6, 0,0,0,0])
        #                            0 1 2 3 4  5 6 7 8
        expected_df = pd.DataFrame({'startBlinks': [0], 'endBlinks': [5]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Blink at start of signal")

    def test_blink_at_end_of_signal(self):
        # Blink duration: 4 to 9 (exclusive) = 5 frames. 5 > 3 -> True
        blink_component = np.array([0,0,0,0, 6,6,6,6,6])
        #                            0 1 2 3  4 5 6 7 8
        expected_df = pd.DataFrame({'startBlinks': [4], 'endBlinks': [9]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Blink at end of signal")

    def test_blink_spanning_entire_signal(self):
        # Blink duration: 0 to 5 (exclusive) = 5 frames. 5 > 3 -> True
        blink_component = np.array([6,6,6,6,6])
        #                            0 1 2 3 4
        expected_df = pd.DataFrame({'startBlinks': [0], 'endBlinks': [5]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Blink spanning entire signal")

    def test_two_valid_blinks_sufficiently_separated(self):
        # Min gap frames = 0.5s * 100Hz = 50 frames
        # Blink 1: 3 to 8 (exclusive), duration 5. Valid.
        # Blink 2: 60 to 65 (exclusive), duration 5. Valid.
        # Gap: Start of blink2 (60) - End of blink1 (8) = 52 frames. 52 > 50. Valid.
        signal_part1 = np.array([0,0,0, 6,6,6,6,6, 0]) # Blink ends at index 8 (exclusive)
        #                           0 1 2  3 4 5 6 7  8
        gap_duration = 52 # frames
        signal_gap = np.zeros(gap_duration)
        signal_part2 = np.array([6,6,6,6,6, 0,0,0])   # Blink starts at index 0 relative to this part
        #                           0 1 2 3 4  5 6 7
        blink_component = np.concatenate((signal_part1, signal_gap, signal_part2))

        start_blink2 = len(signal_part1) + len(signal_gap)
        end_blink2 = start_blink2 + 5

        expected_df = pd.DataFrame({'startBlinks': [3, start_blink2],
                                    'endBlinks': [8, end_blink2]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Two valid, separated blinks")

    def test_two_valid_blinks_too_close(self):
        # Min gap frames = 0.5s * 100Hz = 50 frames
        # Blink 1: 3 to 8 (exclusive), duration 5. Valid.
        # Blink 2: 10 to 15 (exclusive), duration 5. Valid.
        # Gap: Start of blink2 (10) - End of blink1 (8) = 2 frames. 2 <= 50. Too close. Both removed.
        signal_part1 = np.array([0,0,0, 6,6,6,6,6, 0]) # Blink ends at index 8 (exclusive)
        #                           0 1 2  3 4 5 6 7  8
        gap_duration = 1 # frames (so total gap between end of first true and start of second true is 2)
        signal_gap = np.zeros(gap_duration) # This is the one sample at index 8
        signal_part2 = np.array([0, 6,6,6,6,6, 0,0,0]) # Blink starts at index 1 (relative to part2)
        #                        0  1 2 3 4 5  6 7 8

        # Actual structure: [0,0,0, 6,6,6,6,6, 0,  0,  6,6,6,6,6, 0,0,0]
        # Indices:          0,1,2, 3,4,5,6,7, 8,  9, 10,1,2,3,4, 5,6,7
        # Blink 1: starts 3, ends 8 (exclusive)
        # Blink 2: starts 10, ends 15 (exclusive)
        # Gap frames = 10 - 8 = 2.
        # 2 <= 50, so too close. Both should be removed.

        blink_component = np.array([0,0,0, 6,6,6,6,6, 0, 0, 6,6,6,6,6, 0,0,0])

        expected_df = pd.DataFrame({'startBlinks': [], 'endBlinks': []}, dtype=int)
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Two valid blinks, too close")

    def test_one_blink_too_short_one_valid(self):
        # Min gap frames = 50
        # Blink 1 (too short): 3 to 5 (exclusive), duration 2. 2 > 3 -> False. Removed by duration filter.
        # Blink 2 (valid): 60 to 65 (exclusive), duration 5. 5 > 3 -> True. Kept.
        signal_part1 = np.array([0,0,0, 6,6, 0]) # Ends at index 5 (exclusive)
        gap_duration = 55 # > 50
        signal_gap = np.zeros(gap_duration)
        signal_part2 = np.array([6,6,6,6,6, 0,0,0]) # Starts at 0 relative to this part

        blink_component = np.concatenate((signal_part1, signal_gap, signal_part2))

        start_blink2 = len(signal_part1) + len(signal_gap)
        end_blink2 = start_blink2 + 5

        expected_df = pd.DataFrame({'startBlinks': [start_blink2], 'endBlinks': [end_blink2]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: One short, one valid blink")

    def test_three_blinks_middle_one_causes_removal_of_neighbors(self):
        # B1: valid, B2: valid, B3: valid
        # Gap B1-B2: too short
        # Gap B2-B3: too short
        # Result: all removed
        # min_blink_frames = 3 (duration > 3, so at least 4 frames)
        # min_gap_frames = 50
        self.min_blink_frames = 3
        self.params["minEventLen"] = 0.1 # 10 frames gap
        min_gap_frames = self.params["minEventLen"] * self.sfreq # 10 frames

        # Blink: 4 frames (duration 4)
        blink_signal = np.array([6,6,6,6]) # duration 4, valid (4 > 3)
        short_gap_signal = np.zeros(5) # gap = 5 frames. 5 <= 10. Too short.

        # B1 (idx 0-4) --gap1 (5 frames)-- B2 (idx 9-13) --gap2 (5 frames)-- B3 (idx 18-22)
        # B1: start 0, end 4
        # B2: start 4+5=9, end 9+4=13
        # B3: start 13+5=18, end 18+4=22
        # Gap B1-B2: B2.start (9) - B1.end (4) = 5.  5 <= 10 (too_close). Keep[0]=F, Keep[1]=F
        # Gap B2-B3: B3.start (18) - B2.end (13) = 5. 5 <= 10 (too_close). Keep[1]=F, Keep[2]=F
        # All should be removed.

        blink_component = np.concatenate([
            blink_signal, short_gap_signal, # B1, gap1
            blink_signal, short_gap_signal, # B2, gap2
            blink_signal
        ])
        expected_df = pd.DataFrame({'startBlinks': [], 'endBlinks': []}, dtype=int)
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Three blinks, cascading removal due to proximity")


    def test_three_blinks_first_two_too_close_third_ok(self):
        # B1, B2 too close (removed). B3 is fine relative to where B2 *would have been*,
        # but since B2 is removed, B3 is now isolated and should be kept.
        # This tests if the `keep` mask correctly allows non-adjacent blinks after removal.
        self.min_blink_frames = 3 # duration > 3 (at least 4 frames)
        self.params["minEventLen"] = 0.1 # 10 frames gap
        min_gap_frames = self.params["minEventLen"] * self.sfreq # 10 frames

        blink_signal = np.array([6,6,6,6]) # duration 4
        short_gap_signal = np.zeros(5)     # gap = 5 frames (too short)
        long_gap_signal = np.zeros(15)     # gap = 15 frames (sufficient)

        # B1 (0-4) --short_gap (5 fr)-- B2 (9-13) --long_gap (15 fr)-- B3 (28-32)
        # B1: start 0, end 4
        # B2: start 9, end 13
        # B3: start 13+15=28, end 28+4=32

        # Gap B1-B2: 9-4 = 5.  5 <= 10 (too_close). Keep[0]=F, Keep[1]=F
        # Gap B2-B3: 28-13 = 15. 15 > 10 (ok).
        # Since B1 and B2 are removed, B3 should remain.

        blink_component = np.concatenate([
            blink_signal, short_gap_signal, # B1, short_gap1
            blink_signal, long_gap_signal,  # B2, long_gap
            blink_signal                    # B3
        ])

        expected_df = pd.DataFrame({'startBlinks': [28], 'endBlinks': [32]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: B1,B2 too close (removed), B3 kept")

    def test_input_parameter_validation(self):
        blink_component = np.array([1.0])
        with self.assertRaisesRegex(ValueError, "blink_component must be provided"):
            get_blink_position(self.params, None, threshold=self.threshold, min_blink_frames=self.min_blink_frames)
        with self.assertRaisesRegex(ValueError, "threshold and min_blink_frames must be provided"):
            get_blink_position(self.params, blink_component, threshold=None, min_blink_frames=self.min_blink_frames)
        with self.assertRaisesRegex(ValueError, "threshold and min_blink_frames must be provided"):
            get_blink_position(self.params, blink_component, threshold=self.threshold, min_blink_frames=None)
        with self.assertRaisesRegex(KeyError, "params must contain 'sfreq' and 'minEventLen'"):
            get_blink_position({"minEventLen": 0.1}, blink_component, threshold=self.threshold, min_blink_frames=self.min_blink_frames)
        with self.assertRaisesRegex(KeyError, "params must contain 'sfreq' and 'minEventLen'"):
            get_blink_position({"sfreq": 100}, blink_component, threshold=self.threshold, min_blink_frames=self.min_blink_frames)

    def test_signal_ends_mid_blink_before_min_duration(self):
        # Blink starts, but signal ends before min_blink_frames is met
        # min_blink_frames = 3, so needs >3 frames (at least 4)
        blink_component = np.array([0,0, 6,6,6]) # Starts at 2, ends at 5 (exclusive). Duration 3. 3 > 3 is false.
        expected_df = pd.DataFrame({'startBlinks': [], 'endBlinks': []}, dtype=int)
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Signal ends mid-blink, too short")

    def test_signal_ends_mid_blink_after_min_duration(self):
        # Blink starts, signal ends, duration is sufficient
        # min_blink_frames = 3, so needs >3 frames (at least 4)
        blink_component = np.array([0,0, 6,6,6,6,6]) # Starts at 2, ends at 7 (exclusive). Duration 5. 5 > 3 is true.
        expected_df = pd.DataFrame({'startBlinks': [2], 'endBlinks': [7]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Signal ends mid-blink, sufficient duration")

    def test_signal_starts_mid_blink_then_valid_end(self):
        # Signal starts already above threshold
        # min_blink_frames = 3, so needs >3 frames (at least 4)
        blink_component = np.array([6,6,6,6,6, 0,0]) # Starts at 0, ends at 5 (exclusive). Duration 5. 5 > 3 is true.
        expected_df = pd.DataFrame({'startBlinks': [0], 'endBlinks': [5]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Signal starts mid-blink, valid end")

    def test_signal_starts_mid_blink_but_too_short(self):
        # Signal starts already above threshold, but total duration is too short
        # min_blink_frames = 3, so needs >3 frames (at least 4)
        blink_component = np.array([6,6,6, 0,0]) # Starts at 0, ends at 3 (exclusive). Duration 3. 3 > 3 is false.
        expected_df = pd.DataFrame({'startBlinks': [], 'endBlinks': []}, dtype=int)
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Signal starts mid-blink, too short")

    def test_edge_case_short_signal_all_blink_too_short(self):
        # Signal is entirely a blink, but shorter than min_blink_frames + 1
        # min_blink_frames = 3, so needs >3 frames (at least 4)
        blink_component = np.array([7, 8, 7]) # Duration 3. 3 > 3 is False.
        expected_df = pd.DataFrame({'startBlinks': [], 'endBlinks': []}, dtype=int)
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Short signal, all blink, too short")

    def test_edge_case_short_signal_all_blink_just_enough(self):
        # Signal is entirely a blink, and is min_blink_frames + 1 long
        # min_blink_frames = 3, so needs >3 frames (at least 4)
        blink_component = np.array([7, 8, 7, 9]) # Duration 4. 4 > 3 is True.
        expected_df = pd.DataFrame({'startBlinks': [0], 'endBlinks': [4]})
        result_df = get_blink_position(self.params, blink_component,
                                       threshold=self.threshold,
                                       min_blink_frames=self.min_blink_frames)
        self.assert_dataframes_equal(result_df, expected_df, "Test: Short signal, all blink, just enough")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
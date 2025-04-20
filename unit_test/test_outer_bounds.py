"""
test_compute_outer_bounds.py

Unit test and visualization utilities for the compute_outer_bounds function.

This script provides:
- A test case using a mock candidate_signal with known maxFrame peaks.
- Assertions to ensure the function returns expected outerStarts and outerEnds.
- A visualization tool using matplotlib to show how blink intervals
  are computed from shifted maxFrame values.

Key Components:
---------------
1. test_compute_outer_bounds:
   - Creates a candidate_signal with strong peaks at frames [10, 25, 40, 60].
   - Uses those as maxFrame values in a test DataFrame.
   - Applies compute_outer_bounds to verify correct interval calculation.
   - Asserts correctness and visualizes results.

2. plot_outer_bounds:
   - Visualizes the outer bounds as horizontal lines on a frame index axis.
   - Highlights maxFrame points to show where each blink's core activity is.

Usage:
------
Run directly as a script to execute the test and view the plot:
    python test_compute_outer_bounds.py

Or include as part of an automated test suite using pytest.

Dependencies:
-------------
- numpy
- pandas
- matplotlib
- compute_outer_bounds (import from processing_utils)

Author: Your Name
Date: 2025-04-17
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.testing import assert_frame_equal
from pyblinkers.zero_crossing import compute_outer_bounds


def generate_mock_signal():
    """
    Generates a synthetic candidate_signal with known peaks
    at indices 10, 25, 40, 60.
    """
    signal = np.zeros(80)
    signal[10] = 1.0   # Blink peak 1
    signal[25] = 0.9   # Blink peak 2
    signal[40] = 1.1   # Blink peak 3
    signal[60] = 0.95  # Blink peak 4

    # Add some random background noise
    noise_indices = np.setdiff1d(np.arange(80), [10, 25, 40, 60])
    signal[noise_indices] = np.random.normal(loc=0.0, scale=0.05, size=len(noise_indices))
    return signal


def plot_outer_bounds(df, signal):
    """
    Visualizes the blink intervals and signal with maxFrames marked.
    Each interval is labeled (below the line) with its blink index, range, and maxFrame.
    Also shows a vertical red line for each maxFrame and labeled x-ticks.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal, label='candidate_signal', alpha=0.6, color='gray')

    for i, row in df.iterrows():
        start = row['outerStarts']
        end = row['outerEnds']
        maxf = row['maxFrame']

        # Draw horizontal line for blink range
        ax.hlines(y=1.2, xmin=start, xmax=end,
                  color='skyblue', linewidth=3, label='Blink Window' if i == 0 else "")

        # Draw vertical line at maxFrame
        ax.axvline(x=maxf, color='red', linestyle='--', linewidth=1, label='maxFrame line' if i == 0 else "")

        # Mark maxFrame as red dot
        ax.plot(maxf, 1.2, 'o', color='red')

        # Add annotation below line
        annotation = f"Blink {i}: [{int(start)} → {int(end)}], max @{int(maxf)}"
        ax.text((start + end) / 2, 1.05, annotation,
                ha='center', va='top', fontsize=8, color='black', rotation=45)

    # Set x-ticks at regular intervals
    frame_count = len(signal)
    xticks = np.arange(0, frame_count + 1, 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Signal Value")
    ax.set_title("Blink Potential Ranges with maxFrame Indicators")
    ax.set_ylim(-1, 1.6)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()



class TestComputeOuterBounds(unittest.TestCase):

    def test_compute_outer_bounds(self):
        """
        Test compute_outer_bounds using a mock candidate_signal with strong peaks
        at known positions. Ensures outerStarts and outerEnds align with expected
        frame window logic around maxFrames.
        """
        signal = generate_mock_signal()
        data_size = len(signal)

        # Manually define maxFrame peaks
        df = pd.DataFrame({'maxFrame': [10, 25, 40, 60]})

        # Expected DataFrame
        expected = pd.DataFrame({
            'maxFrame': [10, 25, 40, 60],
            'outerStarts': [0, 10, 25, 40],
            'outerEnds': [25, 40, 60, 80]
        })

        # Compute actual result
        result = compute_outer_bounds(df, data_size)

        # Assert correctness
        assert_frame_equal(result, expected)

        # Visualize
        plot_outer_bounds(result, signal)


if __name__ == '__main__':
    unittest.main()

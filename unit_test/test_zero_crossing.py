"""
test_zero_crossing.py

Unit f test and visualization utilities for the left_right_zero_crossing function.

This script provides:
- A test case using mock signal data to validate the behavior of the
  left_right_zero_crossing function under typical conditions.
- Assertions to ensure the function returns expected crossing points.
- A visualization tool using matplotlib to aid in understanding
  how the left and right zero crossings are detected around a given max_blink.

Key Components:
---------------
1. test_left_right_zero_crossing:
   - Creates a mock candidate_signal with a known negative-to-positive
     transition pattern.
   - Tests the left_right_zero_crossing function to ensure it identifies the
     correct last negative index before max_blink (left_zero), and the first
     negative index after max_blink (right_zero).
   - Includes sanity checks and validations via assertions.
   - Optionally visualizes the result using the plot_zero_crossings function.

2. plot_zero_crossings:
   - Takes the signal, frame indices, and highlights the zero crossing
     locations using vertical lines on a matplotlib plot.

Usage:
------
Run directly as a script to execute the test and view the plot:
    python test_zero_crossing.py

Or include as part of an automated test suite using pytest.

Dependencies:
-------------
- numpy
- matplotlib
- left_right_zero_crossing function (import from your main project module)

Author: Your Name
Date: 2025-04-17
"""

"""
test_zero_crossing.py

Unit tests and visualization for the left_right_zero_crossing function.
This script tests detection of zero crossings and provides plotting to debug visually.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyblinkers.zero_crossing import left_right_zero_crossing


def plot_zero_crossings(candidate_signal, max_blink, left_zero, right_zero,
                        outer_start, outer_end, title="Zero Crossing Visualization"):
    """
    Plots the signal using scatter and vertical lines for zero crossing indicators.
    """
    x = np.arange(len(candidate_signal))

    plt.figure(figsize=(12, 5))
    plt.scatter(x, candidate_signal, label='Signal', c='black', s=25)
    plt.axhline(0, color='gray', linestyle='-', linewidth=1)  # zero line

    # Vertical markers
    plt.axvline(outer_start, color='blue', linestyle='--', label='outer_start')
    plt.axvline(outer_end, color='blue', linestyle='--', label='outer_end')
    plt.axvline(max_blink, color='orange', linestyle='--', label='max_blink')

    if left_zero is not None:
        plt.axvline(left_zero, color='green', linestyle='--', label='left_zero')
    if right_zero is not None:
        plt.axvline(right_zero, color='red', linestyle='--', label='right_zero')

    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel("Signal Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_left_right_zero_crossing_basic():
    """Standard case: zero crossings found within left/right outer ranges."""
    candidate_signal = np.array([0.5, -0.4, -0.3, 0.2, 0.4, 0.5, 0.6, -0.1, -0.3, 0.2])
    max_blink = 4
    outer_start = 1
    outer_end = 8

    left_zero, right_zero = left_right_zero_crossing(candidate_signal, max_blink, outer_start, outer_end)
    print(f"Test Basic - Left: {left_zero}, Right: {right_zero}")

    assert left_zero is not None
    assert right_zero is not None
    assert left_zero <= max_blink
    assert max_blink <= right_zero

    plot_zero_crossings(candidate_signal, max_blink, left_zero, right_zero, outer_start, outer_end,
                        title="Test Basic - Normal Zero Crossings")


def test_left_zero_fallback():
    """Triggers fallback: no negative values within outer_start to max_blink."""
    candidate_signal = np.array([-0.2,0.2, 0.3, 0.4, 0.5, -0.1, -0.3])
    max_blink = 3
    outer_start = 1
    outer_end = 5  # won't matter here

    left_zero, right_zero = left_right_zero_crossing(candidate_signal, max_blink, outer_start, outer_end)
    print(f"Test Left Fallback - Left: {left_zero}, Right: {right_zero}")

    assert left_zero is not None, "Fallback should find last negative in [0, max_blink]"
    assert right_zero is not None, "Right zero should not be None"

    plot_zero_crossings(candidate_signal, max_blink, left_zero, right_zero, outer_start, outer_end,
                        title="Test Left Fallback")


def test_right_zero_fallback():
    """Triggers fallback: no negative values within max_blink to outer_end."""
    candidate_signal = np.array([0.2, -0.2, -0.3, 0.4, 0.5, 0.6, 0.7, -0.4])
    max_blink = 5
    outer_start = 0
    outer_end = 6  # no negative between 5 and 6

    left_zero, right_zero = left_right_zero_crossing(candidate_signal, max_blink, outer_start, outer_end)
    print(f"Test Right Fallback - Left: {left_zero}, Right: {right_zero}")

    assert left_zero is not None
    assert right_zero == 7, "Fallback should find first negative in [max_blink, end]"

    plot_zero_crossings(candidate_signal, max_blink, left_zero, right_zero, outer_start, outer_end,
                        title="Test Right Fallback")

def test_left_right_zero_nan():
    """   - If no negative values are found, set left_zero or right_zero to np.nan. This especially true when
    we deal with epoch format, as the signal windows might be small, therefore,we cannot extend the search window
    to extreme."""
    candidate_signal = np.array([0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.7, 0.7])
    max_blink = 5
    outer_start = 0
    outer_end = 6

    left_zero, right_zero = left_right_zero_crossing(candidate_signal, max_blink, outer_start, outer_end)
    print(f"Test Right Fallback - Left: {left_zero}, Right: {right_zero}")

    assert np.isnan(left_zero), "Fallback should give nan for left_zero"
    assert np.isnan(right_zero), "Fallback should give nan for right_zero"


if __name__ == "__main__":
    test_left_right_zero_crossing_basic()
    test_left_zero_fallback()
    test_right_zero_fallback()
    test_left_right_zero_nan()

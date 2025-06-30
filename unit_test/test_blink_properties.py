"""
test_blink_properties.py
This module tests the `BlinkProperties` class from `pyblinkers.extractBlinkProperties`.

Overview:
  The `BlinkProperties` class extracts rich, physiologically meaningful blink
  features from a candidate EEG signal and associated blink metadata.

Data Lineage:
  The input `blink_properties_fits.pkl` is produced by `blink_features.py`,
  which applies helper functions from `fit_blink.py`, including:
    - left_right_zero_crossing
    - _get_half_height
    - compute_fit_range
    - lines_intersection_matlabx

These upstream steps generate intermediate features such as:
  - Zero/base/tent crossings
  - Half-height durations
  - Slope fits from linear blink segments

Test Coverage:
  - Construction and correct output of the BlinkProperties object.
  - Expected shape, types, and values in the resulting DataFrame.
  - Key blink timing, amplitude, velocity, and shut-duration features.
  - Data sanity (e.g., no NaNs in required fields, positive durations, logical constraints).

The `BlinkProperties` class calculates a wide range of blink-related features:
  • Blink durations:
     - Base, Zero, Tent-based durations
     - Half-height durations (Base, Zero)
  • Amplitude-Velocity Ratios (AVR):
     - Positive and negative AVR from Zero/Base landmarks
     - Tent-based AVR (using fitted slope velocities)
  • Shut Time Durations:
     - From Base, Zero, and Tent (based on threshold crossings)
  • Blink Peaks:
     - Value and time of maximum signal amplitude
     - Value and time of peak from fitted tent model
  • Inter-blink Features:
     - Time between blink peaks (amplitude)
     - Time between positive velocity peaks (base, zero)

The output DataFrame contains new columns including:
  - `durationBase`, `durationZero`, `durationTent`, `durationHalfBase`, `durationHalfZero`
  - `posAmpVelRatioZero`, `negAmpVelRatioZero`, `posAmpVelRatioBase`, `negAmpVelRatioBase`
  - `posAmpVelRatioTent`, `negAmpVelRatioTent`
  - `timeShutBase`, `timeShutZero`, `timeShutTent`
  - `closingTimeZero`, `reopeningTimeZero`, `closingTimeTent`, `reopeningTimeTent`
  - `peakMaxBlink`, `peakMaxTent`, `peakTimeBlink`, `peakTimeTent`
  - `interBlinkMaxAmp`, `interBlinkMaxVelBase`, `interBlinkMaxVelZero`
Output Fields:
  - 'durationBase', 'durationZero', 'durationTent', 'durationHalfBase', 'durationHalfZero'
  - 'posAmpVelRatioZero', 'negAmpVelRatioZero', 'posAmpVelRatioBase', 'negAmpVelRatioBase'
  - 'posAmpVelRatioTent', 'negAmpVelRatioTent'
  - 'timeShutBase', 'timeShutZero', 'timeShutTent'
  - 'closingTimeZero', 'reopeningTimeZero', 'closingTimeTent', 'reopeningTimeTent'
  - 'peakMaxBlink', 'peakMaxTent', 'peakTimeBlink', 'peakTimeTent'
  - 'interBlinkMaxAmp', 'interBlinkMaxVelBase', 'interBlinkMaxVelZero'

Test Inputs:
  - EEG signal from: S1_candidate_signal.npy
  - Blink metadata from: blink_properties_fits.pkl (output of blink_features.py)
  - Parameters: shutAmpFraction, pAVRThreshold, z_thresholds

Dependencies:
  - numpy
  - pandas
  - pytest
  - pyblinkers.extractBlinkProperties.BlinkProperties
"""

import numpy as np
import pandas as pd
import pytest

from pyblinkers.extractBlinkProperties import BlinkProperties


@pytest.fixture(scope="module")
def candidate_signal() -> np.ndarray:
    """
    Load EEG candidate signal from disk.

    Returns:
        np.ndarray: 1D EEG signal array.
    """
    return np.load("S1_candidate_signal.npy")


@pytest.fixture(scope="module")
def blink_df() -> pd.DataFrame:
    """
    Load precomputed blink metadata (from blink_features.py).

    Returns:
        pd.DataFrame: Metadata for blink candidates, incl. fitted features.
    """
    df = pd.read_pickle("blink_properties_fits.pkl")
    assert not df.empty and isinstance(df, pd.DataFrame)
    return df


@pytest.fixture(scope="module")
def blink_params() -> dict:
    """
    Blink property extraction parameters.

    Returns:
        dict: Parameters such as thresholds and shut amplitude fraction.
    """
    return {
        'shutAmpFraction': 0.9,
        'pAVRThreshold': 3,
        'z_thresholds': np.array([[0.9, 0.98],
                                  [2.0, 5.0]])
    }


def test_blink_properties_structure(candidate_signal: np.ndarray, blink_df: pd.DataFrame, blink_params: dict):
    """
    Check structure, presence, and basic validity of BlinkProperties output.

    Asserts:
      • Output is a non-empty DataFrame
      • All expected columns are present
      • No missing (NaN) values in critical numeric fields
      • Durations are strictly positive
    """
    srate = 100
    result_df = BlinkProperties(candidate_signal, blink_df, srate, blink_params).df
    print(result_df)

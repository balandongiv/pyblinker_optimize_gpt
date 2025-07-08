"""Helper for median absolute deviation similar to MATLAB."""
from __future__ import annotations

import numpy as np


def mad_matlab(arr: np.ndarray, axis: int | None = None, keepdims: bool = True) -> np.ndarray:
    """Compute median absolute deviation similar to MATLAB."""
    median = np.median(arr, axis=axis, keepdims=True)
    mad = np.median(np.abs(arr - median), axis=axis, keepdims=keepdims)[0]
    return mad


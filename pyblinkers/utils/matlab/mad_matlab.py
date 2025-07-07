"""Helper for median absolute deviation similar to MATLAB."""
from __future__ import annotations

import numpy as np


def mad_matlab(arr: np.ndarray, axis: int | None = None, keepdims: bool = True) -> np.ndarray:
    """Return the median absolute deviation of *arr* mimicking MATLAB behavior.

    Parameters
    ----------
    arr:
        Input array of numeric values.
    axis:
        Axis along which to compute the MAD. Defaults to the flattened array.
    keepdims:
        Whether to keep the reduced dimension in the output.

    Returns
    -------
    numpy.ndarray
        The median absolute deviation of ``arr``.
    """
    median = np.median(arr, axis=axis, keepdims=True)
    mad = np.median(np.abs(arr - median), axis=axis, keepdims=keepdims)
    if keepdims:
        return mad[0]
    return mad


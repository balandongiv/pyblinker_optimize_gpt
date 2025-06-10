from typing import List, Tuple

import numpy as np


def _ensure_peaks_positive(sig: np.ndarray) -> np.ndarray:
    """If overall maximum is negative, flip the entire signal."""
    if np.nanmax(sig) < 0:
        sig = -sig
    return sig

def refine_local_mean(sig: np.ndarray, epochs: List[Tuple[int, int]], **_) -> np.ndarray:
    """Subtract mean of *each epoch* independently (baseline wander removal)."""
    out = sig.copy()
    for start, end in epochs:
        out[start:end] -= np.mean(out[start:end])
    return _ensure_peaks_positive(out)
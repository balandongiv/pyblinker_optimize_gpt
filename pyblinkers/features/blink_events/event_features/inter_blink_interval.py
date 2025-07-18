"""Inter-blink interval based features."""
from __future__ import annotations

from typing import Dict, List, Sequence

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _permutation_entropy(series: Sequence[float], *, order: int = 3, delay: int = 1) -> float:
    """Calculate permutation entropy of a numeric sequence.

    Parameters
    ----------
    series : Sequence[float]
        Input data sequence representing inter-blink intervals.
    order : int, optional
        Embedding dimension for ordinal pattern creation, by default ``3``.
    delay : int, optional
        Sample delay between points in each pattern, by default ``1``.

    Returns
    -------
    float
        Estimated permutation entropy. Returns ``NaN`` if the sequence is too
        short for the specified parameters.
    """
    data = np.asarray(series)
    n = len(data)
    if n < order * delay:
        return float("nan")
    patterns: List[tuple[int, ...]] = []
    for i in range(n - delay * (order - 1)):
        subseq = data[i : i + order * delay : delay]
        patterns.append(tuple(np.argsort(subseq)))
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / counts.sum()
    pe = -np.sum(probs * np.log(probs))
    return float(pe)


def _hurst_exponent(series: Sequence[float]) -> float:
    """Estimate the Hurst exponent from a sequence using R/S analysis.

    Parameters
    ----------
    series : Sequence[float]
        Sequence of inter-blink intervals.

    Returns
    -------
    float
        Hurst exponent estimating long-range dependence. ``NaN`` is returned for
        very short or constant sequences.
    """
    data = np.asarray(series, dtype=float)
    n = len(data)
    if n < 20:
        return float("nan")
    mean = np.mean(data)
    dev = data - mean
    cumdev = np.cumsum(dev)
    r = np.max(cumdev) - np.min(cumdev)
    s = np.std(data, ddof=1)
    if s == 0 or r == 0:
        return float("nan")
    return float(np.log(r / s) / np.log(n))


def compute_ibi_features(blinks: List[Dict[str, int]], sfreq: float) -> Dict[str, float]:
    """Compute inter-blink interval statistics for a given epoch.

    Parameters
    ----------
    blinks : list of dict
        Blink annotations belonging to one epoch.
    sfreq : float
        Sampling frequency of the original recording in Hertz.

    Returns
    -------
    dict
        Dictionary with summary metrics of inter-blink intervals including mean,
        standard deviation and complexity measures.
    """
    starts = np.array([b["refined_start_frame"] for b in blinks], dtype=float)
    ends = np.array([b["refined_end_frame"] for b in blinks], dtype=float)
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]
    ibis: np.ndarray | None = None
    if len(starts) >= 2:
        ibis = (starts[1:] - ends[:-1]) / sfreq
    if ibis is None or len(ibis) == 0:
        return {
            "ibi_mean": float("nan"),
            "ibi_std": float("nan"),
            "ibi_median": float("nan"),
            "ibi_min": float("nan"),
            "ibi_max": float("nan"),
            "ibi_cv": float("nan"),
            "ibi_rmssd": float("nan"),
            "poincare_sd1": float("nan"),
            "poincare_sd2": float("nan"),
            "poincare_ratio": float("nan"),
            "ibi_permutation_entropy": float("nan"),
            "ibi_hurst_exponent": float("nan"),
        }

    ibi_mean = float(np.mean(ibis))
    ibi_std = float(np.std(ibis, ddof=1)) if len(ibis) > 1 else float("nan")
    ibi_median = float(np.median(ibis))
    ibi_min = float(np.min(ibis))
    ibi_max = float(np.max(ibis))
    ibi_cv = float(ibi_std / ibi_mean) if ibi_mean != 0 else float("nan")
    diff = np.diff(ibis)
    rmssd = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else float("nan")
    if len(ibis) > 2:
        x1 = ibis[:-1]
        x2 = ibis[1:]
        sd1 = float(np.sqrt(np.var(x2 - x1, ddof=1) / 2.0))
        sd2 = float(np.sqrt(np.var(x1 + x2, ddof=1) / 2.0))
        sd_ratio = float(sd1 / sd2) if sd2 != 0 else float("nan")
    else:
        sd1 = sd2 = sd_ratio = float("nan")
    pe = _permutation_entropy(ibis)
    hurst = _hurst_exponent(ibis)
    return {
        "ibi_mean": ibi_mean,
        "ibi_std": ibi_std,
        "ibi_median": ibi_median,
        "ibi_min": ibi_min,
        "ibi_max": ibi_max,
        "ibi_cv": ibi_cv,
        "ibi_rmssd": rmssd,
        "poincare_sd1": sd1,
        "poincare_sd2": sd2,
        "poincare_ratio": sd_ratio,
        "ibi_permutation_entropy": pe,
        "ibi_hurst_exponent": hurst,
    }

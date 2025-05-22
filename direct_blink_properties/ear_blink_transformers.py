"""
blink_refiners.py
=================

This module defines a suite of signal refinement functions tailored to blink detection
in eye-aspect-ratio (EAR) time-series. Each function transforms the raw EAR signal
to enhance blink epoch detection by emphasizing positive peaks and reducing baseline noise.

====================
Capabilities
------------

**Signal Refinement**
All refiners produce a transformed signal of the same length as the input, with the
goal of making each blink interval produce a distinct **positive peak**. Some methods
are global, others operate on local epochs.

**Caching**
Results are automatically cached in `.cache/*.pkl` using a stable MD5 hash of the method
name and parameter dictionary. When re-running, results are silently loaded unless
inputs or configurations have changed.

**Metrics & Ranking**
Each refinement is evaluated using ground-truth blink epochs and scored on:

- `kept_ratio`: Proportion of blinks retained after filtering invalid zero-crossings.
- `positive_peak_fraction`: Fraction of blink intervals where the refined signal has a max > 0.
- `median_peak_amp`: Median amplitude of the blink peak across valid epochs.

These metrics are computed in `_evaluate()` and stored in `Metrics`. A leaderboard is
returned as a ranked `DataFrame` (higher = better), sorted by:
    kept_ratio → positive_peak_fraction → median_peak_amp

**Structured Logging**
Pipeline progress and evaluations are tracked using Python’s `logging` module.
Verbose results can be enabled by increasing the logging level (e.g., DEBUG).

=================
Signal‑refinement strategies Overview
---------------
Each method below corresponds to a registered refiner. Most support parameter grids
for exhaustive benchmarking.

Basic Methods:
--------------
- `global_mean`: Subtracts the global mean and flips sign to enforce positive peaks.
- `local_mean`: Subtracts the mean within each blink epoch (requires `epochs`).
- `detrend_linear`: Removes a best-fit linear trend from the entire signal.
- `detrend_poly`: Polynomial trend removal using least-squares (orders 2–4).
- `savgol`: Savitzky-Golay filter for local polynomial baseline smoothing.
- `robust_scale`: Applies robust scaling using median-centered normalization.

Filters:
--------
- `highpass`: Butterworth high-pass filter (e.g. cutoff 0.2 Hz).
- `highpass_iir`: IIR Chebyshev-II high-pass filter (zero-phase).
- `bandpass`: Butterworth band-pass filter (e.g. 0.1–10 Hz).
- `firhp`: Kaiser-window FIR high-pass filter.

Smoothing / Trend Removal:
--------------------------
- `als`: Asymmetric Least Squares baseline correction (configurable `lam` and `p`).
- `airpls`: Adaptive iteratively reweighted penalized least squares (PLS).
- `whittaker`: Smoother using Whittaker penalized least squares.
- `medfilt`: Rolling median filter.
- `quantile`: Rolling quantile baseline subtraction (e.g. 5th percentile).
- `loess`: Locally weighted scatterplot smoothing (LOWESS).
- `zscore`: Rolling z-score normalization.
- `tophat`: Morphological top-hat transform using opening.
- `morph`: Morphological opening followed by closing.

Decomposition-Based:
--------------------
- `wavelet`: Multi-resolution wavelet transform (removes approximation coefficients).
- `emd`: Empirical Mode Decomposition – sum of early intrinsic mode functions (IMFs).
- `ceemdan`: CEEMDAN (noise-assisted EMD) – sum of selected IMFs.
- `vmd`: Variational Mode Decomposition – sum of dominant modes.

Epoch-Aware:
------------
- `epoch_poly`: Polynomial detrending applied per epoch with boundary padding.
- `quantile_hp`: Cascade of rolling quantile removal + high-pass filter.
- `quantile_hp_flip`: Same as `quantile_hp`, but with epoch-wise peak polarity correction.

Meta-Techniques:
- `_ensure_peaks_positive`: Ensures max > 0 in global signal.
- `apply_epoch_flip`: Ensures each blink epoch has a positive peak.


Each method ensures output signal length matches the input and guarantees positive blink peaks.
The `run_experiments()` function benchmarks these over multiple configurations with metrics:
`kept_ratio`, `positive_peak_fraction`, and `median_peak_amp`.

See Also:
---------
- `_ensure_peaks_positive`: Utility to flip signal globally if needed.
- `apply_epoch_flip`: Epoch-level polarity enforcement.
- `left_right_zero_crossing`: Core detection logic (unchanged from base code).
- `_evaluate`: Core metric computation.

The core blink‑segmentation routine `left_right_zero_crossing` is *copied verbatim*
from your code and must remain untouched.

Usage example
-------------
# >>> from blink_signal_refinement import run_experiments
# >>> leaderboard = run_experiments(raw_signal, blink_df, params)
# >>> print(leaderboard.head())

You can also run the module directly (`python blink_signal_refinement.py`) – it will
create a synthetic demo signal if no FIF/zip path is supplied.

==============================
Choosing the Best Method

The leaderboard DataFrame ranks each refiner configuration using the formula:
sort by (kept_ratio DESC, positive_peak_fraction DESC, median_peak_amp DESC)

This means that the method which retains the most blink intervals with valid zero-crossings and
ensures strong positive peaks will appear at the top.

The top-ranked row is typically the most robust method, but you may also inspect
the trade-off between kept_ratio and median_peak_amp if your application prefers
signal clarity over detection breadth.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pywt
import statsmodels.api as sm
from PyEMD import CEEMDAN
from PyEMD import EMD
from scipy import sparse
from scipy.ndimage import grey_closing
from scipy.ndimage import grey_opening
from scipy.signal import butter
from scipy.signal import detrend, savgol_filter
from scipy.signal import iirfilter, firwin, filtfilt
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from vmdpy import VMD

from direct_blink_properties.blink_epoch_normalization import refine_local_mean

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("blink")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
logger.addHandler(_handler)

_CACHE_DIR = Path(".cache")
_CACHE_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Helper – hashing utils for caching
# -----------------------------------------------------------------------------

def _hash(obj) -> str:
    """Stable md5 hash of any picklable object."""
    return hashlib.md5(pickle.dumps(obj, protocol=4)).hexdigest()

# -----------------------------------------------------------------------------
# Left / right zero‑crossing – **do not modify**
# -----------------------------------------------------------------------------

def _to_ints(*args):
    return [int(a) for a in args]


def left_right_zero_crossing(candidate_signal: np.ndarray,
                             max_frame: int,
                             outer_starts: int,
                             outer_ends: int) -> Tuple[float, float]:
    """Copied unchanged from user code."""
    start_idx, end_idx, m_frame = _to_ints(outer_starts, outer_ends, max_frame)

    # Left side search
    left_range = np.arange(start_idx, m_frame)
    left_values = candidate_signal[left_range]
    s_ind_left_zero = np.flatnonzero(left_values < 0)
    if s_ind_left_zero.size > 0:
        left_zero = left_range[s_ind_left_zero[-1]]
    else:
        full_left_range = np.arange(0, m_frame).astype(int)
        left_neg_idx = np.flatnonzero(candidate_signal[full_left_range] < 0)
        left_zero = full_left_range[left_neg_idx[-1]] if left_neg_idx.size else np.nan

    # Right side search
    right_range = np.arange(m_frame, end_idx)
    right_values = candidate_signal[right_range]
    s_ind_right_zero = np.flatnonzero(right_values < 0)
    if s_ind_right_zero.size > 0:
        right_zero = right_range[s_ind_right_zero[0]]
    else:
        extreme_outer = np.arange(m_frame, candidate_signal.shape[0]).astype(int)
        s_ind_right_zero_ex = np.flatnonzero(candidate_signal[extreme_outer] < 0)
        right_zero = extreme_outer[s_ind_right_zero_ex[0]] if s_ind_right_zero_ex.size else np.nan

    return left_zero, right_zero

# -----------------------------------------------------------------------------
# Dataclass containers
# -----------------------------------------------------------------------------

@dataclass
class Metrics:
    kept_ratio: float  # rows kept / rows original
    positive_peak_fraction: float  # proportion epochs whose max > 0
    median_peak_amp: float  # median of blink‑max amplitudes

    def to_dict(self):
        return asdict(self)


@dataclass
class ExperimentResult:
    method: str
    params: Dict
    metrics: Metrics

    def to_row(self):
        d = self.metrics.to_dict()
        d.update(method=self.method)
        return d


# -----------------------------------------------------------------------------
# Signal refinement functions – each returns the *same‑length* 1‑D array
# -----------------------------------------------------------------------------

def refine_tophat(sig: np.ndarray, size: int = 401, **_) -> np.ndarray:
    """Morphological opening (top-hat) baseline removal."""
    base = grey_opening(sig, size=size)
    return _ensure_peaks_positive(sig - base)


def refine_highpass_iir(sig: np.ndarray, fs: float, fc: float = 0.2, order: int = 4, **_) -> np.ndarray:
    """Chebyshev-II IIR high-pass filter, zero-phase."""
    b, a = iirfilter(order, fc/(fs/2), btype='high', ftype='cheby2', rp=1, rs=40)
    return _ensure_peaks_positive(filtfilt(b, a, sig))


def refine_firhp(sig: np.ndarray, fs: float, fc: float = 0.25, width: float = 0.15, ripple: float = 60, **_) -> np.ndarray:
    """Kaiser-window FIR high-pass filter."""
    # design
    taps = firwin(ripple, fc, window=('kaiser', ripple), pass_zero=False, fs=fs)
    return _ensure_peaks_positive(filtfilt(taps, [1.0], sig))


def refine_ceemdan(sig: np.ndarray, keep_imfs: int = 4, **_) -> np.ndarray:
    """CEEMDAN decomposition, sum of first modes."""
    imfs = CEEMDAN().ceemdan(sig)
    return _ensure_peaks_positive(imfs[:keep_imfs].sum(axis=0))


def refine_vmd(sig: np.ndarray, K: int = 4, alpha: float = 2000, **_) -> np.ndarray:
    """Variational Mode Decomposition baseline fixer."""
    u, _, _ = VMD(sig, K=K, DC=0, init=1, alpha=alpha)
    return _ensure_peaks_positive(u[:K].sum(axis=0))


def refine_epoch_poly(sig: np.ndarray, epochs: List[Tuple[int,int]], order: int = 2, pad: int = 200, **_) -> np.ndarray:
    """Epoch-wise polynomial detrend with padding."""
    out = sig.copy()
    for s, e in epochs:
        a = max(0, s-pad)
        b = min(sig.size, e+pad)
        x = np.arange(a, b)
        p = np.polyfit(x, out[a:b], order)
        out[a:b] = out[a:b] - np.polyval(p, x)
    return _ensure_peaks_positive(out)


def refine_global_mean(sig: np.ndarray, **_) -> np.ndarray:
    """Global mean centring + sign flip to make peaks positive."""
    logger.debug("Global mean centring …")
    sig = -sig  # Flip so blinks trend positive (your original heuristic)
    sig -= np.mean(sig)
    return _ensure_peaks_positive(sig)





def refine_highpass(sig: np.ndarray, fs: float, cutoff: float = 0.25,
                    order: int = 4, **_) -> np.ndarray:
    b, a = butter(order, cutoff / (fs / 2), btype="highpass")
    return _ensure_peaks_positive(filtfilt(b, a, sig))

def refine_bandpass(sig: np.ndarray, fs: float, low: float = 0.1,
                    high: float = 15, order: int = 4, **_) -> np.ndarray:
    b, a = butter(order, [low / (fs/2), high / (fs/2)], btype="band")
    return _ensure_peaks_positive(filtfilt(b, a, sig))
def _als_baseline(y, lam=1e5, p=0.01, niter=10):

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D @ D.T
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def _airpls(y, lam=1e5, niter=15):


    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = spdiags(w, 0, L, L)
        z = spsolve(W + lam * D @ D.T, w * y)
        w = np.exp(-(y - z) / (np.std(y - z) + 1e-8))
        w[y > z] = 1
    return z

def refine_airpls(sig: np.ndarray, lam: float = 1e5, **_) -> np.ndarray:
    return _ensure_peaks_positive(sig - _airpls(sig, lam))


def refine_wavelet(sig: np.ndarray, wave: str = "db4", level: int = 6, **_) -> np.ndarray:
    coeffs = pywt.wavedec(sig, wave, mode="symmetric")
    coeffs[0] *= 0  # kill the lowest-frequency approximation
    return _ensure_peaks_positive(pywt.waverec(coeffs, wave, mode="symmetric"))




def _whittaker(y, lam=1600):
    m = len(y)
    E = diags([1, -2, 1], [0, -1, -2], shape=(m-2, m))
    return y - spsolve(E.T @ E + lam * sparse.eye(m), y)

def refine_whittaker(sig: np.ndarray, lam: float = 1600, **_) -> np.ndarray:
    return _ensure_peaks_positive(_whittaker(sig, lam))


def refine_medfilt(sig: np.ndarray, win: int = 501, **_) -> np.ndarray:
    baseline = pd.Series(sig).rolling(win, center=True, min_periods=1).median().values
    return _ensure_peaks_positive(sig - baseline)

def refine_quantile(sig: np.ndarray, win: int = 601, q: float = 0.05, **_) -> np.ndarray:
    baseline = pd.Series(sig).rolling(win, center=True, min_periods=1).quantile(q).values
    return _ensure_peaks_positive(sig - baseline)




def refine_morph(sig: np.ndarray, size: int = 201, **_) -> np.ndarray:
    base = grey_opening(grey_closing(sig, size=size), size=size)
    return _ensure_peaks_positive(sig - base)




def refine_loess(sig: np.ndarray, frac: float = 0.02, **_) -> np.ndarray:
    baseline = sm.nonparametric.lowess(sig, np.arange(sig.size), frac=frac, return_sorted=False)
    return _ensure_peaks_positive(sig - baseline)



def refine_emd(sig: np.ndarray, keep_imfs: int = 3, **_) -> np.ndarray:
    imfs = EMD(spline_kind="linear").emd(sig)
    hi = imfs[:keep_imfs].sum(axis=0)
    return _ensure_peaks_positive(hi)


def refine_zscore(sig: np.ndarray, win: int = 301, **_) -> np.ndarray:
    s = pd.Series(sig)
    mu = s.rolling(win, center=True, min_periods=1).mean()
    sd = s.rolling(win, center=True, min_periods=1).std().replace(0, np.nan)
    return _ensure_peaks_positive((s - mu) / sd.fillna(1)).to_numpy()


def refine_als(sig: np.ndarray, lam: float = 1e5, p: float = 0.01, **_) -> np.ndarray:
    return _ensure_peaks_positive(sig - _als_baseline(sig, lam, p))

# -------------------------------------------------------------------------
# Meta-techniques: epoch flip, derivative check, dynamic gap, cascade
# -------------------------------------------------------------------------

def apply_epoch_flip(sig: np.ndarray,
                     epochs: List[Tuple[int,int]]) -> np.ndarray:
    """Flip any epoch whose max ≤ 0 so it becomes positive."""
    out = sig.copy()
    for s, e in epochs:
        if out[s:e].max() <= 0:
            out[s:e] *= -1
    return out



def refine_quantile_hp(sig: np.ndarray, fs: float,
                       win: int = 601, cutoff: float = 0.15) -> np.ndarray:
    """
    Often, a cascade of two simple refiners (e.g. rolling-quantile → highpass) is more powerful than either alone.

    Two-stage baseline removal + high-pass cascade.

    1) Rolling-quantile baseline removal:
       subtract the 5th percentile over a sliding window of `win` samples
    2) 4th-order Butterworth high-pass filter with cutoff `cutoff` Hz
    3) Ensure that the global maximum is positive by flipping the entire signal if needed

    Parameters
    ----------
    sig : np.ndarray
        Raw 1-D signal.
    fs : float
        Sampling frequency in Hz.
    win : int, optional
        Window length (in samples) for rolling quantile, by default 601.
    cutoff : float, optional
        Cutoff frequency (Hz) for the high-pass filter, by default 0.15.

    Returns
    -------
    np.ndarray
        Refined signal with positive peaks.
    """
    # 1) quantile baseline
    sig1 = refine_quantile(sig, win=win, q=0.05)

    # 2) butter-highpass
    sig2 = refine_highpass(sig1, fs=fs, cutoff=cutoff,order=4)

    return _ensure_peaks_positive(sig2)


def refine_quantile_hp_epoch_flip(sig: np.ndarray,
                            fs: float,
                            epochs: List[Tuple[int,int]],
                            win: int = 601,
                            q: float = 0.05,
                            cutoff: float = 0.15) -> np.ndarray:
    """
    Three‐stage cascade with per‐epoch polarity correction.

    Compared to `refine_quantile_hp`, this adds an explicit epoch‐wise flip pass
    so that *every* blink window ends up with a positive peak.

    Steps:
      1. Rolling‐quantile baseline removal
         - Compute the qth percentile (default 5th) over a sliding window of `win` samples,
           and subtract it to remove slow baseline drift.
      2. 4th‐order Butterworth high‐pass filter
         - Zero‐phase (filtfilt) at `cutoff` Hz to remove remaining low‐frequency content.
      3. Epoch‐wise polarity flip
         - For each blink epoch defined in `epochs` (start, end indices),
           if the local maximum amplitude is non‐positive, flip that window (`*=-1`),
           guaranteeing a positive peak in every window.
      4. Global peak check
         - Finally, if the *entire* signal’s maximum is negative (rare), flip globally.

    Parameters
    ----------
    sig : np.ndarray
        Original 1‐D EAR signal.
    fs : float
        Sampling frequency in Hz.
    epochs : List[Tuple[int,int]]
        List of (start, end) sample indices for each blink epoch.
    win : int, optional
        Sliding window length (samples) for computing the rolling quantile (baseline),
        by default 601.
    q : float, optional
        Quantile to subtract (e.g. 0.05 for 5th‐percentile baseline), by default 0.05.
    cutoff : float, optional
        High‐pass cutoff frequency in Hz, by default 0.15.

    Returns
    -------
    np.ndarray
        Refined signal where every blink epoch has a clearly positive peak.
    """
    # 1) quantile baseline removal
    sig1 = refine_quantile(sig, win=win, q=q)

    # 2) butter‐highpass

    b, a = butter(N=4, Wn=cutoff/(fs/2), btype="highpass")
    sig2 = filtfilt(b, a, sig1)

    # 3) per‐epoch polarity flip
    out = sig2.copy()
    for s, e in epochs:
        if out[s:e].max() <= 0:
            out[s:e] *= -1

    # 4) global peak check
    return _ensure_peaks_positive(out)




def refine_detrend_linear(sig: np.ndarray, **_) -> np.ndarray:
    return _ensure_peaks_positive(detrend(sig, type="linear"))


def refine_detrend_poly(sig: np.ndarray, order: int = 3, **_) -> np.ndarray:
    logger.debug("Polynomial detrend order=%d", order)
    x = np.arange(sig.size)
    coeffs = np.polyfit(x, sig, order)
    trend = np.polyval(coeffs, x)
    return _ensure_peaks_positive(sig - trend)


def refine_savgol(sig: np.ndarray, window: int = 301, polyorder: int = 3, **_) -> np.ndarray:
    logger.debug("Savitzky‑Golay baseline window=%d order=%d", window, polyorder)
    baseline = savgol_filter(sig, window_length=window, polyorder=polyorder)
    return _ensure_peaks_positive(sig - baseline)


def refine_robust(sig: np.ndarray, **_) -> np.ndarray:
    scaler = RobustScaler(with_centering=True, with_scaling=False)
    sig2 = scaler.fit_transform(sig.reshape(-1, 1)).ravel()
    return _ensure_peaks_positive(sig2)


# Registry for convenience -----------------------------------------------------



# -----------------------------------------------------------------------------
# Utility – force positive blink peaks (epoch max > 0)
# -----------------------------------------------------------------------------

def _ensure_peaks_positive(sig: np.ndarray) -> np.ndarray:
    """If overall maximum is negative, flip the entire signal."""
    if np.nanmax(sig) < 0:
        sig = -sig
    return sig

# -----------------------------------------------------------------------------
# Core: run one refinement + blink evaluation (with caching)
# -----------------------------------------------------------------------------

def _evaluate(refined: np.ndarray, df: pd.DataFrame,params:dict) -> Metrics:
    """Attach L/R zero‑crossings, drop invalid rows and compute metrics."""
    original_len = len(df)

    tmp = df.copy()
    tmp[["leftZero", "rightZero"]] = tmp.apply(
        lambda row: left_right_zero_crossing(
            refined, row["blinkmin"], row["startBlinks"], row["endBlinks"]
        ),
        axis=1,
        result_type="expand",
    )

    tmp = tmp.dropna()
    min_gap = max(3, int(params["sfreq"] * 0.01))  # e.g. 0.01 s = 10 ms
    keep = (tmp["rightZero"] - tmp["leftZero"]) >= min_gap
    tmp = tmp[keep]
    # tmp = tmp[(tmp["rightZero"] - tmp["leftZero"]) >= 3]
    kept_len = len(tmp)

    # Metrics
    peaks = refined[tmp["blinkmin"].astype(int)]
    positive_peak_fraction = np.mean(peaks > 0) if peaks.size else 0.0
    median_peak_amp = np.median(peaks) if peaks.size else 0.0

    metrics = Metrics(
        kept_ratio=kept_len / original_len if original_len else 0.0,
        positive_peak_fraction=float(positive_peak_fraction),
        median_peak_amp=float(median_peak_amp),
    )

    logger.info("Evaluation result: kept_ratio=%.3f | positive_peak_fraction=%.3f | median_peak_amp=%.3f",
                metrics.kept_ratio, metrics.positive_peak_fraction, metrics.median_peak_amp)

    return metrics


def _cache_load(method: str, params: Dict) -> Metrics | None:
    key = f"{method}_{_hash(params)}"
    path = _CACHE_DIR / f"{key}.pkl"
    if path.exists():
        metrics: Metrics = pickle.loads(path.read_bytes())
        logger.info(
            "Loaded cached result for %s → kept=%.3f | positive=%.3f | median_amp=%.3f",
            method, metrics.kept_ratio, metrics.positive_peak_fraction, metrics.median_peak_amp
        )
        return metrics
    return None


def _cache_save(method: str, params: Dict, metrics: Metrics):
    key = f"{method}_{_hash(params)}"
    path = _CACHE_DIR / f"{key}.pkl"
    path.write_bytes(pickle.dumps(metrics, protocol=4))
    logger.debug("Saved cache → %s", path)


# -----------------------------------------------------------------------------
# Public API – run experiments & return leaderboard (pd.DataFrame)
# -----------------------------------------------------------------------------

def run_experiments(candidate_signal: np.ndarray,
                    df: pd.DataFrame,
                    params: Dict,
                    REFINERS: Dict[str, Callable],
                    grids: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Run every registered refiner (and param grid, if any) → leaderboard."""
    logger.info("Starting experiments on %d samples – %d blink intervals",
                candidate_signal.size, len(df))

    # REFINERS: Dict[str, Callable] = {
    #     "global_mean": refine_global_mean,
    #     "local_mean": refine_local_mean,
    #     "detrend_linear": refine_detrend_linear,
    #     "detrend_poly": refine_detrend_poly,
    #     "savgol": refine_savgol,
    #     "robust_scale": refine_robust,
    #     "highpass": refine_highpass,
    #     "bandpass": refine_bandpass,
    #     "als": refine_als,
    #     "airpls": refine_airpls,
    #     "wavelet": refine_wavelet,
    #     "whittaker": refine_whittaker,
    #     "medfilt": refine_medfilt,
    #     "quantile": refine_quantile,
    #     "morph": refine_morph,
    #     "loess": refine_loess,
    #     "emd": refine_emd,
    #     "zscore": refine_zscore,
    #     "tophat": refine_tophat,
    #     "highpass_iir": refine_highpass_iir,
    #     "firhp": refine_firhp,
    #     "ceemdan": refine_ceemdan,
    #     "vmd": refine_vmd,
    #     "epoch_poly": refine_epoch_poly,
    #     "quantile_hp":refine_quantile_hp,
    #     "quantile_hp_flip":refine_quantile_hp_epoch_flip
    #
    # }

    # REFINERS: Dict[str, Callable] = {
    #     # Not yet run
    #     "loess": refine_loess,
    #     "emd": refine_emd,
    #     "zscore": refine_zscore,
    #     "tophat": refine_tophat,
    #     "highpass_iir": refine_highpass_iir,
    #     "firhp": refine_firhp,
    #     "ceemdan": refine_ceemdan,
    #     "vmd": refine_vmd,
    #     "epoch_poly": refine_epoch_poly,
    #     "quantile_hp": refine_quantile_hp,
    #     "quantile_hp_flip": refine_quantile_hp_epoch_flip,
    #
    #     # Run, sorted by kept descending
    #     "quantile": refine_quantile,           # kept=1.000
    #     "highpass": refine_highpass,           # kept=0.959
    #     "medfilt": refine_medfilt,             # kept=0.685
    #     "als": refine_als,                     # kept=0.877
    #     "wavelet": refine_wavelet,             # kept=0.863
    #     "local_mean": refine_local_mean,       # kept=0.849
    #     "airpls": refine_airpls,               # kept=0.603
    #     "detrend_poly": refine_detrend_poly,   # kept=0.534
    #     "detrend_linear": refine_detrend_linear,  # kept=0.479
    #     "savgol": refine_savgol,               # kept=0.384
    #     "global_mean": refine_global_mean,     # kept=0.397
    #     "robust_scale": refine_robust,         # kept=0.288
    #     "morph": refine_morph,                 # kept=0.247
    #     "bandpass": refine_bandpass,           # kept=0.000
    #     "whittaker": refine_whittaker          # kept=0.000
    # }

    # Build simple parameter grid – can be expanded as needed
    # epochs = list(zip(df.startBlinks, df.endBlinks))
    # sfreq = params["sfreq"]

    # grids: Dict[str, List[Dict]] = {
    #     # Basic methods
    #     "global_mean": [{}],
    #     "local_mean": [{"epochs": epochs}],
    #     "detrend_linear": [{}],
    #     "detrend_poly": [{"order": o} for o in (2, 3, 4)],
    #     "savgol": [{"window": w, "polyorder": 3} for w in (151, 301, 601)],
    #     "robust_scale": [{}],
    #
    #     # Filters
    #     "highpass": [{"fs": sfreq, "cutoff": c} for c in (0.1, 0.2, 0.3)],
    #     "highpass_iir": [{"fs": sfreq, "fc": fc} for fc in (0.15, 0.2, 0.25)],
    #     "bandpass": [{"fs": sfreq, "low": 0.1, "high": h} for h in (10, 15)],
    #     "firhp": [{"fs": sfreq, "fc": 0.25, "width": 0.1}],
    #
    #     # Smoothing / trend removal
    #     "als": [{"lam": l, "p": p} for l in (1e4, 1e5) for p in (0.01, 0.05)],
    #     "airpls": [{"lam": l} for l in (1e4, 1e5, 1e6)],
    #     "whittaker": [{"lam": l} for l in (800, 1600, 3200)],
    #     "medfilt": [{"win": w} for w in (301, 501, 801)],
    #     "quantile": [{"win": w, "q": 0.05} for w in (401, 601)],
    #     "loess": [{"frac": f} for f in (0.01, 0.02, 0.03)],
    #     "zscore": [{"win": w} for w in (201, 301, 401)],
    #     "tophat": [{"size": s} for s in (301, 401, 501)],
    #     "morph": [{"size": s} for s in (101, 201, 301)],
    #
    #     # Decomposition-based
    #     "wavelet": [{"wave": "db4", "level": lv} for lv in (4, 5, 6)],
    #     "emd": [{"keep_imfs": k} for k in (2, 3, 4)],
    #     "ceemdan": [{"keep_imfs": k} for k in (3, 4, 5)],
    #     "vmd": [{"K": k, "alpha": a} for k in (3, 4) for a in (1500, 2000)],
    #
    #     # Epoch-aware
    #     "epoch_poly": [{"epochs": epochs, "order": o} for o in (1, 2)],
    #     "quantile_hp": [
    #         {"fs": sfreq, "win": w, "cutoff": c}
    #         for w in (401, 601, 801)
    #         for c in (0.10, 0.15, 0.20)
    #     ],
    #     "quantile_hp_flip": [
    #         {"fs": sfreq, "epochs": epochs, "win": w, "q": 0.05, "cutoff": c}
    #         for w in (401, 601, 801)
    #         for c in (0.1, 0.15, 0.2)
    #     ]
    # }




    joblist = [
            (method, fn, p)
            for method, fn in REFINERS.items()
            for p in grids[method]
        ]

    results: List[ExperimentResult] = []

    # create a standalone progress bar
    with tqdm(total=len(joblist), desc="Running refiners", unit="method") as pbar:
        for method, fn, p in joblist:
            try:
                # 1) load or compute
                cached = _cache_load(method, p)
                if cached is None:
                    logger.info("%s – running…", method)
                    refined = fn(candidate_signal.copy(), **p)
                    metrics = _evaluate(refined, df, params)
                    _cache_save(method, p, metrics)
                else:
                    metrics = cached

                # 2) store result
                results.append(ExperimentResult(method, p, metrics))

                # 3) update bar and postfix
                pbar.set_postfix(method=method, kept=f"{metrics.kept_ratio:.3f}")

            except Exception as e:
                logger.warning("Skipping %s due to error: %s", method, str(e))

            finally:
                pbar.update(1)


    leaderboard = pd.DataFrame([r.to_row() for r in results])
    leaderboard.sort_values(["kept_ratio", "positive_peak_fraction", "median_peak_amp"],
                            ascending=False, inplace=True)
    logger.info("Experiments finished – best method: %s",
                leaderboard.iloc[0]["method"])
    return leaderboard.reset_index(drop=True)

def get_refiners() -> Dict[str, Callable]:
    return {
        # Not yet run
        # "loess": refine_loess,
        # "emd": refine_emd,
        "zscore": refine_zscore,
        "tophat": refine_tophat,
        "highpass_iir": refine_highpass_iir,
        "firhp": refine_firhp,
        # "ceemdan": refine_ceemdan,
        # "vmd": refine_vmd,
        "epoch_poly": refine_epoch_poly,
        "quantile_hp": refine_quantile_hp,
        "quantile_hp_flip": refine_quantile_hp_epoch_flip,
        #
        # # Run, sorted by kept descending
        "quantile": refine_quantile,           # kept=1.000
        "highpass": refine_highpass,           # kept=0.959
        "medfilt": refine_medfilt,             # kept=0.685
        # "als": refine_als,                     # kept=0.877
        # "wavelet": refine_wavelet,             # kept=0.863
        "local_mean": refine_local_mean,       # kept=0.849
        "airpls": refine_airpls,               # kept=0.603
        # "detrend_poly": refine_detrend_poly,   # kept=0.534
        # "detrend_linear": refine_detrend_linear,  # kept=0.479
        # "savgol": refine_savgol,               # kept=0.384
        "global_mean": refine_global_mean,     # kept=0.397
        # "robust_scale": refine_robust,         # kept=0.288
        # "morph": refine_morph,                 # kept=0.247
        "bandpass": refine_bandpass,           # kept=0.000
        # "whittaker": refine_whittaker          # kept=0.000
    }

def get_grids(df: pd.DataFrame, sfreq: float) -> Dict[str, List[Dict]]:
    epochs = list(zip(df.startBlinks, df.endBlinks))
    return {
        # Basic methods
        "global_mean": [{}],
        "local_mean": [{"epochs": epochs}],
        "detrend_linear": [{}],
        "detrend_poly": [{"order": o} for o in (2, 3, 4)],
        "savgol": [{"window": w, "polyorder": 3} for w in (151, 301, 601)],
        "robust_scale": [{}],

        # Filters
        "highpass": [{"fs": sfreq, "cutoff": c} for c in (0.1, 0.2, 0.3)],
        "highpass_iir": [{"fs": sfreq, "fc": fc} for fc in (0.15, 0.2, 0.25)],
        "bandpass": [{"fs": sfreq, "low": 0.1, "high": h} for h in (10, 15)],
        "firhp": [{"fs": sfreq, "fc": 0.25, "width": 0.1}],

        # Smoothing / trend removal
        "als": [{"lam": l, "p": p} for l in (1e4, 1e5) for p in (0.01, 0.05)],
        "airpls": [{"lam": l} for l in (1e4, 1e5, 1e6)],
        "whittaker": [{"lam": l} for l in (800, 1600, 3200)],
        "medfilt": [{"win": w} for w in (301, 501, 801)],
        "quantile": [{"win": w, "q": 0.05} for w in (401, 601)],
        "loess": [{"frac": f} for f in (0.01, 0.02, 0.03)],
        "zscore": [{"win": w} for w in (201, 301, 401)],
        "tophat": [{"size": s} for s in (301, 401, 501)],
        "morph": [{"size": s} for s in (101, 201, 301)],

        # Decomposition-based
        "wavelet": [{"wave": "db4", "level": lv} for lv in (4, 5, 6)],
        "emd": [{"keep_imfs": k} for k in (2, 3, 4)],
        "ceemdan": [{"keep_imfs": k} for k in (3, 4, 5)],
        "vmd": [{"K": k, "alpha": a} for k in (3, 4) for a in (1500, 2000)],

        # Epoch-aware
        "epoch_poly": [{"epochs": epochs, "order": o} for o in (1, 2)],
        "quantile_hp": [
            {"fs": sfreq, "win": w, "cutoff": c}
            for w in (401, 601, 801)
            for c in (0.10, 0.15, 0.20)
        ],
        "quantile_hp_flip": [
            {"fs": sfreq, "epochs": epochs, "win": w, "q": 0.05, "cutoff": c}
            for w in (401, 601, 801)
            for c in (0.1, 0.15, 0.2)
        ]
    }
# -----------------------------------------------------------------------------
# CLI helper for quick manual tests
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    fif_path = Path(r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.fif")
    zip_path = Path(r"C:\Users\balan\IdeaProjects\pyblinker_optimize_gpt\data_new_pipeline\S01_20170519_043933.zip")

    # Optional parameters
    use_data_files = fif_path.exists() and zip_path.exists()

    if use_data_files:
        from direct_blink_properties.util import load_fif_and_annotations, extract_blink_durations

        raw, annotation_df = load_fif_and_annotations(fif_path, zip_path)
        sfreq = raw.info["sfreq"]
        video_fps = 30
        frame_offset = 5
        blink_df = extract_blink_durations(annotation_df, frame_offset, sfreq, video_fps)
        signal = raw.get_data(picks=1)[0]
        params_demo = {"sfreq": sfreq}

    params_demo = {"sfreq": sfreq}
    refiners = get_refiners()
    grids = get_grids(blink_df, sfreq)
    board = run_experiments(signal, blink_df, params_demo, refiners, grids)
    print("\nLeaderboard (higher = better):")
    print(board.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

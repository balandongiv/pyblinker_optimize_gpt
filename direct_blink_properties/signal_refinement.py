import logging
from typing import List, Tuple

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
from vmdpy import VMD

# Initialize the logger to use the "blink" logger configured in the main script
logger = logging.getLogger("blink")
# -----------------------------------------------------------------------------
# Signal refinement functions – each returns the *same‑length* 1‑D array
# -----------------------------------------------------------------------------
def _ensure_peaks_positive(sig: np.ndarray) -> np.ndarray:
    """If overall maximum is negative, flip the entire signal."""
    if np.nanmax(sig) < 0:
        logger.debug("Signal maximum is negative, flipping signal.")
        sig = -sig
    return sig
def refine_tophat(sig: np.ndarray, size: int = 401, **_) -> np.ndarray:
    """Morphological opening (top-hat) baseline removal."""
    logger.debug(f"Applying refine_tophat with size={size}")
    base = grey_opening(sig, size=size)
    return _ensure_peaks_positive(sig - base)


def refine_highpass_iir(sig: np.ndarray, fs: float, fc: float = 0.2, order: int = 4, **_) -> np.ndarray:
    """Chebyshev-II IIR high-pass filter, zero-phase."""
    logger.debug(f"Applying refine_highpass_iir with fs={fs:.2f}, fc={fc:.2f}, order={order}")
    b, a = iirfilter(order, fc/(fs/2), btype='high', ftype='cheby2', rp=1, rs=40)
    return _ensure_peaks_positive(filtfilt(b, a, sig))


def refine_firhp(sig: np.ndarray, fs: float, fc: float = 0.25, width: float = 0.15, ripple: float = 60, **_) -> np.ndarray:
    """Kaiser-window FIR high-pass filter."""
    # design
    logger.debug(f"Applying refine_firhp with fs={fs:.2f}, fc={fc:.2f}, width={width:.2f}, ripple={ripple}")
    taps = firwin(ripple, fc, window=('kaiser', ripple), pass_zero=False, fs=fs)
    return _ensure_peaks_positive(filtfilt(taps, [1.0], sig))


def refine_ceemdan(sig: np.ndarray, keep_imfs: int = 4, **_) -> np.ndarray:
    """CEEMDAN decomposition, sum of first modes."""
    logger.debug(f"Applying refine_ceemdan, keeping {keep_imfs} IMFs")
    imfs = CEEMDAN().ceemdan(sig)
    return _ensure_peaks_positive(imfs[:keep_imfs].sum(axis=0))


def refine_vmd(sig: np.ndarray, K: int = 4, alpha: float = 2000, **_) -> np.ndarray:
    """Variational Mode Decomposition baseline fixer."""
    logger.debug(f"Applying refine_vmd with K={K}, alpha={alpha}")
    u, _, _ = VMD(sig, K=K, DC=0, init=1, alpha=alpha)
    return _ensure_peaks_positive(u[:K].sum(axis=0))


def refine_epoch_poly(sig: np.ndarray, epochs: List[Tuple[int,int]], order: int = 2, pad: int = 200, **_) -> np.ndarray:
    """Epoch-wise polynomial detrend with padding."""
    logger.debug(f"Applying refine_epoch_poly with order={order}, pad={pad} for {len(epochs)} epochs")
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
    logger.debug("Applying refine_global_mean: Global mean centring and sign flip.")
    sig = -sig  # Flip so blinks trend positive (your original heuristic)
    sig -= np.mean(sig)
    return _ensure_peaks_positive(sig)


def refine_highpass(sig: np.ndarray, fs: float, cutoff: float = 0.25,
                    order: int = 4, **_) -> np.ndarray:
    b, a = butter(order, cutoff / (fs / 2), btype="highpass")
    return _ensure_peaks_positive(filtfilt(b, a, sig))

def refine_bandpass(sig: np.ndarray, fs: float, low: float = 0.1,
                    high: float = 15, order: int = 4, **_) -> np.ndarray:

    logger.debug(f"Applying refine_highpass (Butterworth) with fs={fs:.2f}, cutoff={cutoff:.2f}, order={order}")
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
    logger.debug(f"Applying refine_airpls with lam={lam}")
    return _ensure_peaks_positive(sig - _airpls(sig, lam))


def refine_wavelet(sig: np.ndarray, wave: str = "db4", level: int = 6, **_) -> np.ndarray:
    logger.debug(f"Applying refine_wavelet with wave='{wave}', level={level}")
    coeffs = pywt.wavedec(sig, wave, mode="symmetric")
    coeffs[0] *= 0  # kill the lowest-frequency approximation
    return _ensure_peaks_positive(pywt.waverec(coeffs, wave, mode="symmetric"))




def _whittaker(y, lam=1600):
    m = len(y)
    E = diags([1., -2., 1.], [0, -1, -2], shape=(m-2, m), format='csr')
    # Original code was: return y - spsolve(E.T @ E + lam * sparse.eye(m), y)
    # This implies E.T @ E is (m, m-2) @ (m-2, m) = (m,m)
    # This calculates the smoothed signal z. The baseline is y - z.
    # So, y - (y-z) = z. The function is returning the residual, not the baseline.
    # The function is y - baseline = y - (y-z_smooth) = z_smooth (if z_smooth is baseline)
    # If spsolve(..., y) gives z_smooth (the baseline):
    z_smooth = spsolve(E.T @ E + lam * sparse.eye(m), y)
    return y - z_smooth # Return residual: signal - baseline

def refine_whittaker(sig: np.ndarray, lam: float = 1600, **_) -> np.ndarray:
    logger.debug(f"Applying refine_whittaker with lam={lam}")
    return _ensure_peaks_positive(_whittaker(sig, lam))


def refine_medfilt(sig: np.ndarray, win: int = 501, **_) -> np.ndarray:
    logger.debug(f"Applying refine_medfilt with window={win}")
    baseline = pd.Series(sig).rolling(win, center=True, min_periods=1).median().values
    return _ensure_peaks_positive(sig - baseline)

def refine_quantile(sig: np.ndarray, win: int = 601, q: float = 0.05, **_) -> np.ndarray:
    logger.debug(f"Applying refine_quantile with window={win}, q={q:.2f}")
    baseline = pd.Series(sig).rolling(win, center=True, min_periods=1).quantile(q).values
    return _ensure_peaks_positive(sig - baseline)




def refine_morph(sig: np.ndarray, size: int = 201, **_) -> np.ndarray:
    logger.debug(f"Applying refine_morph with size={size}")
    base = grey_opening(grey_closing(sig, size=size), size=size)
    return _ensure_peaks_positive(sig - base)




def refine_loess(sig: np.ndarray, frac: float = 0.02, **_) -> np.ndarray:
    logger.debug(f"Applying refine_loess with frac={frac:.2f}")
    baseline = sm.nonparametric.lowess(sig, np.arange(sig.size), frac=frac, return_sorted=False)
    return _ensure_peaks_positive(sig - baseline)



def refine_emd(sig: np.ndarray, keep_imfs: int = 3, **_) -> np.ndarray:
    logger.debug(f"Applying refine_emd, keeping {keep_imfs} IMFs")
    emd_obj = EMD(spline_kind="linear")
    imfs = emd_obj.emd(sig)
    if imfs.shape[0] < keep_imfs:
        logger.warning(f"EMD produced only {imfs.shape[0]} IMFs, requested {keep_imfs}. Using all available IMFs.")
        keep_imfs = imfs.shape[0]
    hi = imfs[:keep_imfs].sum(axis=0)
    return _ensure_peaks_positive(hi)


def refine_zscore(sig: np.ndarray, win: int = 301, **_) -> np.ndarray:
    logger.debug(f"Applying refine_zscore with window={win}")
    s = pd.Series(sig)
    mu = s.rolling(win, center=True, min_periods=1).mean()
    sd = s.rolling(win, center=True, min_periods=1).std().replace(0, np.nan)
    # fillna(1) for sd means if std is 0 or NaN, (s-mu)/1 = s-mu.
    # This might be problematic if mu is also NaN for initial points.
    # A common practice is to fill NaN std with a small epsilon or use global std.
    # Or, ensure min_periods for std is large enough to get a non-NaN value.
    # For now, respecting original logic.
    return _ensure_peaks_positive((s - mu) / sd.fillna(1)).to_numpy()


def refine_als(sig: np.ndarray, lam: float = 1e5, p: float = 0.01, **_) -> np.ndarray:
    logger.debug(f"Applying refine_als with lam={lam}, p={p:.2f}")
    return _ensure_peaks_positive(sig - _als_baseline(sig, lam, p))

# -------------------------------------------------------------------------
# Meta-techniques: epoch flip, derivative check, dynamic gap, cascade
# -------------------------------------------------------------------------

def apply_epoch_flip(sig: np.ndarray,
                     epochs: List[Tuple[int,int]]) -> np.ndarray:
    """Flip any epoch whose max ≤ 0 so it becomes positive."""
    logger.debug(f"Applying epoch flip for {len(epochs)} epochs.")
    out = sig.copy()
    num_flipped = 0
    for s, e in epochs:
        if e > s: # Ensure epoch is valid (end > start)
            epoch_slice = out[s:e]
            if epoch_slice.size > 0 and np.nanmax(epoch_slice) <= 0:
                out[s:e] *= -1
                num_flipped += 1
        else:
            logger.warning(f"Skipping invalid epoch ({s}, {e}) in apply_epoch_flip")

    if num_flipped > 0:
        logger.debug(f"Flipped {num_flipped} out of {len(epochs)} epochs to ensure positive peaks.")
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
    logger.debug(f"Applying refine_quantile_hp cascade: win={win}, cutoff={cutoff:.2f}, fs={fs:.2f}")
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
    g(f"Applying refine_quantile_hp_epoch_flip cascade: win={win}, q={q:.2f}, cutoff={cutoff:.2f}, fs={fs:.2f}, {len(epochs)} epochs")
    # 1) quantile baseline removal
    sig1 = refine_quantile(sig, win=win, q=q)

    # 2) butter‐highpass

    b, a = butter(N=4, Wn=cutoff/(fs/2), btype="highpass")
    sig2 = filtfilt(b, a, sig1)

    # 3) per‐epoch polarity flip
    out = apply_epoch_flip(sig2, epochs) # Will log

    # 4) global peak check
    return _ensure_peaks_positive(out)




def refine_detrend_linear(sig: np.ndarray, **_) -> np.ndarray:
    logger.debug("Applying refine_detrend_linear")
    return _ensure_peaks_positive(detrend(sig, type="linear"))


def refine_detrend_poly(sig: np.ndarray, order: int = 3, **_) -> np.ndarray:
    logger.debug(f"Applying refine_detrend_poly with order={order}")
    x = np.arange(sig.size)
    # Ensure enough points for polyfit
    if sig.size <= order:
        logger.warning(f"Skipping polyfit in refine_detrend_poly due to insufficient points ({sig.size}) for order {order}. Returning original signal.")
        return _ensure_peaks_positive(sig)
    coeffs = np.polyfit(x, sig, order)
    trend = np.polyval(coeffs, x)
    return _ensure_peaks_positive(sig - trend)

def refine_savgol(sig: np.ndarray, window: int = 301, polyorder: int = 3, **_) -> np.ndarray:
    logger.debug(f"Applying refine_savgol: Savitzky‑Golay baseline window={window} order={polyorder}")
    if sig.size < window:
        logger.warning(f"Signal size ({sig.size}) is smaller than window size ({window}). Adjusting window size for Savitzky-Golay filter.")
        # Adjust window to be odd and <= sig.size
        window = min(window, sig.size)
        if window % 2 == 0:
            window -= 1
        if window <= polyorder: # or window < 1 after adjustment
            logger.error(f"Cannot apply Savitzky-Golay: adjusted window {window} is too small for polyorder {polyorder}. Returning original signal.")
            return _ensure_peaks_positive(sig)

    baseline = savgol_filter(sig, window_length=window, polyorder=polyorder)
    return _ensure_peaks_positive(sig - baseline)


def refine_robust(sig: np.ndarray, **_) -> np.ndarray:
    logger.debug("Applying refine_robust (RobustScaler with_centering=True, with_scaling=False)")
    # RobustScaler expects 2D array
    if sig.ndim == 1:
        sig_reshaped = sig.reshape(-1, 1)
    else:
        sig_reshaped = sig

    scaler = RobustScaler(with_centering=True, with_scaling=False)
    sig2 = scaler.fit_transform(sig_reshaped).ravel()
    return _ensure_peaks_positive(sig2)
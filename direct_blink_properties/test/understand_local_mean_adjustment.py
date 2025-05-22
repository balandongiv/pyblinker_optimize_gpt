import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from direct_blink_properties.blink_epoch_normalization import refine_local_mean


def simulate_and_plot_blink_refinement():
    """
    Simulate a negatively-biased eye aspect ratio (EAR) signal with synthetic blink events,
    apply local mean refinement to correct baseline drift within blink epochs, and visualize
    the results.

    Purpose:
    --------
    This function demonstrates how the `refine_local_mean` function transforms a fully negative
    EAR signal to ensure each blink epoch is baseline-aligned and exhibits a positive peak.

    Key Steps:
    ----------
    1. Generate a synthetic EAR signal with sinusoidal drift and added random noise.
    2. Inject synthetic blink events using Hanning-windowed spikes.
    3. Shift the entire signal below zero to simulate real-world scenarios where baseline drift
       makes blinks appear negative.
    4. Apply `refine_local_mean` to perform per-epoch baseline correction and enforce positive peaks.
    5. Plot two vertically-stacked subplots:
       - Top: Original signal (all negative).
       - Bottom: Original vs refined signal with blink epochs highlighted.

    Returns:
    --------
    None. This function shows plots directly.
    """
    # Generate synthetic EAR signal
    n = 1000
    np.random.seed(0)
    x = np.linspace(0, 10, n)
    signal = 0.5 * np.sin(0.5 * x) + 0.05 * np.random.randn(n)

    # Add synthetic blink spikes
    blink_epochs: List[Tuple[int, int]] = [(100, 130), (400, 430), (700, 730)]
    for start, end in blink_epochs:
        signal[start:end] += np.hanning(end - start) * 1.5

    # Shift signal downward to ensure it's fully negative
    signal -= (np.max(signal) + 0.01)

    # Apply baseline correction using local mean refinement
    refined = refine_local_mean(signal, blink_epochs)

    # Plot original and refined signal
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top plot: original signal only
    ax_top.plot(signal, label="Original", color="black")
    ax_top.set_title("Original Signal (All Negative)")
    ax_top.set_ylabel("Amplitude")
    ax_top.grid(True)

    # Bottom plot: original vs refined
    ax_bottom.plot(signal, label="Original", color="black")
    ax_bottom.plot(refined, linestyle='--', label="Refined", color="green")
    for start, end in blink_epochs:
        ax_bottom.axvspan(start, end, alpha=0.3, color='orange')

    ax_bottom.set_title("Original vs Refined Signal with Blink Epochs")
    ax_bottom.set_xlabel("Sample Index")
    ax_bottom.set_ylabel("Amplitude")
    ax_bottom.legend()
    ax_bottom.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_and_plot_blink_refinement()

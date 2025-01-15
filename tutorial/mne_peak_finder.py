import matplotlib.pyplot as plt
import numpy as np

def plot_local_extrema(x_vals, signal_data):
    """Plot local minimums and maximums of a given signal.

    Args:
    x_vals (numpy.ndarray): Array of x values for the signal.
    signal_data (numpy.ndarray): Array of y values for the signal.

    Returns:
    None
    """
    # detection of local minimums and maximums

    local_min = (np.diff(np.sign(np.diff(signal_data))) > 0).nonzero()[0] + 1
    local_max = (np.diff(np.sign(np.diff(signal_data))) < 0).nonzero()[0] + 1

    # plot
    plt.figure(figsize=(12, 5))
    plt.plot(x_vals, signal_data, color='grey')
    plt.plot(x_vals[local_min], signal_data[local_min], "o", label="min", color='r')
    plt.plot(x_vals[local_max], signal_data[local_max], "o", label="max", color='b')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # example data with peaks
    x_vals = np.linspace(-1, 3, 1000)
    signal_data = -0.1 * np.cos(12 * x_vals) + np.exp(-(1 - x_vals) ** 2)

    plot_local_extrema(x_vals, signal_data)
import numpy as np
import pandas as pd

def generate_mock_blink_epoch_data(num_rows=5, seed=42):
    np.random.seed(seed)
    data = {
        'startBlinks': np.random.randint(0, 300, num_rows),
        'endBlinks': np.random.randint(1, 301, num_rows),
        'maxValue': np.random.uniform(1e-6, 1e-5, num_rows),
        'maxFrame': np.random.randint(0, 300, num_rows),
        'outerStarts': np.random.randint(0, 300, num_rows),
        'outerEnds': np.random.randint(0, 300, num_rows),
        'leftZero': np.random.randint(0, 300, num_rows),
        'rightZero': np.random.randint(1, 301, num_rows),
        'maxPosVelFrame': np.random.randint(0, 300, num_rows),
        'maxNegVelFrame': np.random.randint(0, 300, num_rows),
        'leftR2': np.random.uniform(0, 1, num_rows),
        'rightR2': np.random.uniform(0, 1, num_rows),
        'xIntersect': np.random.uniform(0, 300, num_rows),
        'yIntersect': np.random.uniform(-1e-5, 1e-5, num_rows),
        'leftXIntercept': np.random.uniform(0, 300, num_rows),
        'rightXIntercept': np.random.uniform(0, 300, num_rows),
        'xLineCross_l': [np.nan] * num_rows,
        'yLineCross_l': [np.nan] * num_rows,
        'xLineCross_r': [np.nan] * num_rows,
        'yLineCross_r': [np.nan] * num_rows,
    }
    return pd.DataFrame(data)

def generate_mock_blink_signal(length=301, blink_indices=[50, 120, 200, 250], seed=None):
    if seed is not None:
        np.random.seed(seed)
    signal = np.random.normal(loc=0.0, scale=0.05, size=length)
    for idx in blink_indices:
        if idx < length:
            signal[idx] = np.random.uniform(0.8, 1.2)  # strong blink peak
    return signal

# Example usage
if __name__ == "__main__":
    df_list = [generate_mock_blink_epoch_data(num_rows=5, seed=seed) for seed in [42, 43]]
    signal_list = [generate_mock_blink_signal(length=301, blink_indices=[50, 120, 200, 250], seed=seed) for seed in [42, 43]]
    from pyblinkers.extractBlinkProperties import get_blink_statistic_epoch_aggregated
    zThresholds = [
        (0.90, 0.98),  # first threshold set
        (2.00, 5.00)   # second threshold set
    ]
    get_blink_statistic_epoch_aggregated(df_list, zThresholds, signal_list=signal_list)
    print(df_list[0].head())
    print(signal_list[0][:10])  # show first 10 samples of the first signal
